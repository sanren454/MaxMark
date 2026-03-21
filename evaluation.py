import argparse
import sys
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import os
import scipy
import torch.nn as nn
# from modified_stable_diffusion import ModifiedStableDiffusionPipeline
import PIL
from PIL import Image, ImageFilter,ImageEnhance
import cv2
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
from diffusers import AutoencoderKL
#导入自定义的INN Watermark
from newWm_v4 import *
# from newWm_v5 import *

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from util.utils import *
from util.log_utils import *


"""评估使用自定义耦合层的INN模型"""


def log_to_file(file_path, unet, adapter, length, model_path, acc, acc_low, acc_high, margin):
    with open(file_path, 'a') as f:
        f.write(f'\nunet: {unet}\n')
        f.write(f'ft_id : {adapter}\n')
        f.write(f'length : {length}\n')
        f.write(f'model : {model_path}\n')
        f.write(f'bit acc percentage: {acc * 100:.2f}%\n')
        f.write(f'acc_low : {acc_low*100:.2f}%\n')
        f.write(f'acc_high: {acc_high*100:.2f}%\n')
        f.write(f'margin: {margin}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--dataset', default='/data3/changxuanhang/datasets/Gustavosta-Stable-Diffusion-Prompts/')
    parser.add_argument('--model_path', default='/data3/changxuanhang/SD-base-models/SD-v1-5/')
    parser.add_argument('--latent_shape',default=(4,64,64))
    parser.add_argument('--ft_id',default=None,type=str)
    parser.add_argument('--adapter_id',default=None,type=str)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--secret_length', default=48, type=int)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--guidancescale', default=7.5, type=float)
    parser.add_argument('--reverse_inference_steps', default=50, type=int)
    parser.add_argument('--place_mode',default='PLACE_SEQUENTIAL')
    parser.add_argument('--threshold',default=1e-6,type=float)
    parser.add_argument('--model', default='/data3/changxuanhang/watermark/capacity/INN/new_wm/pretrain_inn/test_old/16384_10.0/pretrained_inn_PLACE_SEQUENTIAL_epoch299.pth', type=str)
    parser.add_argument('--birghtness', default=None, type=float,choices=[1,2,3,4,5])
    parser.add_argument('--noise', default=None, type=float,choices=[0.01,0.05])
    parser.add_argument('--contrast', default=None, type=float,choices=[1,2,3,4,5])
    parser.add_argument('--hue', default=None, type=float,choices=[0.25,2])
    parser.add_argument('--blur', default=None, type=int,choices=[1,3,5])
    parser.add_argument('--jpegcompression', default=None, type=int,choices=[40,50])
    parser.add_argument('--resize', default=None, type=float,choices=[0.4,0.8])
    parser.add_argument('--save',type=str,default=None)
    parser.add_argument('--margin',type=float,default=1.0)
    args =parser.parse_known_args()[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_printoptions(sci_mode=False,profile='full')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    maxlength=250
    
    # dataset
    dataset, prompt_key = get_dataset(args)
    dataset=promptdataset(dataset,prompt_key)
    
    # save path 
    save_path = f'eva_RS_paper/{args.save}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    #model
    if(args.ft_id is not None):
        unet =UNet2DConditionModel.from_pretrained(args.ft_id, subfolder="unet", torch_dtype=torch.float16)
        scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder='scheduler')
        pipe = DiffusionPipeline.from_pretrained(
                args.model_path,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision='fp16',
                unet=unet,
                )
    if args.adapter_id is not None:
        pipe = DiffusionPipeline.from_pretrained(args.model_path,
                                                torch_dtype=torch.float16
                                                )
        pipe.load_lora_weights(args.adapter_id)
        pipe.fuse_lora()

    if args.ft_id is None and args.adapter_id is None:
        scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder='scheduler')
        pipe = DiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float32,
            # revision='fp16',
        )

    # if "stable-diffusion-xl-base-1.0" in args.model_path:
    #     vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    #     pipe = DiffusionPipeline.from_pretrained(
    #         args.model_path,
    #         vae=vae,
    #         torch_dtype=torch.float16,
    #         # revision='fp16',
    #         # use_safetensors=False, 
    #         )



    pipe.safety_checker = None
    pipe = pipe.to(device)

    


    # 引入INN
    inn =newWatermark(args.latent_shape).inn.to(device)
    if args.model !=None:
        inn.load_state_dict(torch.load(args.model))
    else:
        raise ValueError("必须加载INN")
    inn.eval()


    with torch.no_grad():
        ACC = []
        img_list = []
        all_error_proportions = []
        for t in tqdm(range(60)):
            secret, positions, inn_input_latent = embed_secret_in_latent(args.secret_length,args.latent_shape,args.place_mode,args.margin) 
            secret = torch.tensor(secret).to(device)
            inn_input_latent = torch.tensor(inn_input_latent).to(device).unsqueeze(0)

            z, log_jac_det = inn(inn_input_latent)

            #对latent手动标准化
            mu = z.mean(dim=(0,2,3),keepdim=True)
            std = z.std(dim=(0,2,3),keepdim=True)
            z_norm = (z-mu)/(std+1e-6)

            # vis_plt = visualize_latent(z,bin_wid=0.1,latent_shape=args.latent_shape)
            # vis_plt.savefig(f'{save_path}/latent_distribution.png')
            # vis_inn_input = visualize_latent(inn_input_latent,bin_wid=0.1,latent_shape=args.latent_shape)
            # vis_inn_input.savefig(f'{save_path}/inn_input_distribution.png')
            # vis_plt_znorm = visualize_latent(z_norm,bin_wid=0.1,latent_shape=args.latent_shape)
            # vis_plt_znorm.savefig(f'{save_path}/latent_norm_distribution.png')
            # sys.exit()
            # prompt=dataset[random.randint(1, len(dataset))][0:maxlength]
            # prompt=dataset[-t-1][0:maxlength]
            prompt = 'rusty warship dreadnought shipwreck in a lush forest, volumetric lighting, god rays, , global illumination, puddles of water, sci-fi.'
            # prompt=dataset[3999][0:maxlength]
            print(f"current prompt: {prompt}")
            print(f'margin={args.margin}')
            img1= pipe(prompt=prompt,num_inference_steps=args.num_inference_steps,\
            latents=z_norm,guidance_scale=args.guidancescale).images[0]
            
            img1.save(f'{save_path}/sup-warship.png')
            sys.exit()
            # if args.birghtness != None:
            #             img1 = transforms.ColorJitter(brightness=args.birghtness)(img1)
            # if args.noise != None:
            #             img1 = np.array(img1, dtype=np.uint8)
            #             g_noise = np.random.randn(*img1.shape).astype(np.uint8) * args.noise
            #             noisy_array = np.clip(img1.astype(np.float32) + g_noise, 0, 255).astype(np.uint8)
            #             img1 = Image.fromarray(noisy_array)
            # if args.contrast != None:
            #             enhancer = ImageEnhance.Contrast(img1)
            #             factor = args.contrast
            #             img1= enhancer.enhance(factor)
            # if args.hue != None:
            #             enhancer = ImageEnhance.Color(img1)
            #             factor = args.hue
            #             img1 = enhancer.enhance(factor)
            # if args.jpegcompression != None:
            #             img1=compress_jpeg_to_pil(img1, args.jpegcompression)
            #             # img1.show()
            # if args.blur != None:
            #             img1=Image.fromarray(cv2.GaussianBlur(np.array(img1),(args.blur,args.blur), 1))
            # if args.resize != None:
            #             img1 = img1.resize((int(args.image_length*args.resize), int(args.image_length*args.resize)), PIL.Image.BICUBIC)

            img_list.append(img1)
            if len(img_list)==5:
                img_stitched = stitch_images(img_list)
                img_stitched.save(f'{save_path}/{t-4}-{t}.png')
                img_list.clear()

            reverse_latents=reverse(img1,pipe,args).float()

            # mean = reverse_latents.mean()
            # std = reverse_latents.std()
            # min_value = reverse_latents.min()
            # max_value = reverse_latents.max()
            # print(f"Mean: {mean.item()}")
            # print(f"Std: {std.item()}")
            # print(f"Min: {min_value.item()}")
            # print(f"Max: {max_value.item()}")
            # sys.exit()

            

            print(f'>>> reverse_latents shape ={reverse_latents.shape}')
            decoded, _ = inn(reverse_latents, rev=True)
            decoded = decoded.flatten()[positions]
            # print(f'>>> secret: \n {secret}')
            # print(f'>>> decoded:\n {decoded}')
            # print(f'>>> secret shape = {secret.shape}')
            # print(f'>>> decoded shape = {decoded.shape}')
            decoded_threshold = (decoded >= 0).int()

            is_similar = torch.abs(decoded_threshold - secret) < args.threshold
            correct_bits = torch.sum(is_similar).item()    
            print(f'>>> correct_bits = {correct_bits}')
            acc = correct_bits/args.secret_length
            ACC.append(acc)
            print('>>> acc = {:.6f}'.format(acc))

            # 计算错误位置的脆弱性分布
            err_indices = torch.nonzero(is_similar == False, as_tuple=True)
            err_indices = [tensor.cpu().numpy() for tensor in err_indices]
            # 将 NumPy 数组堆叠在一起，形成形状为 (n, dim) 的数组
            # 其中 n 是索引数量，dim 是维度数
            err_indices = np.column_stack(err_indices)

            err_por = err_indices/args.secret_length
            all_error_proportions.append(err_por)

            # sys.exit()
    acc = sum(ACC)/len(ACC)
    ACC_sorted = sorted(ACC)
    log_to_file('eva_log.txt',args.ft_id, args.adapter_id,args.secret_length, args.model,acc,ACC_sorted[0],ACC_sorted[-1],args.margin)
    all_error_proportions = np.concatenate(all_error_proportions)
    plot_error_density(f'{save_path}/{args.secret_length}_err_porportions.png',all_error_proportions,args.secret_length)



    