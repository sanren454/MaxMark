import argparse
import sys
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import os, sys
import scipy
import torch.nn as nn
# from modified_stable_diffusion import ModifiedStableDiffusionPipeline
import PIL
from PIL import Image, ImageFilter,ImageEnhance
import cv2
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
#导入自定义的INN Watermark
from newWm_v4 import *

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from util.utils import *
from util.ecc_utils import *
from util.log_utils import *

def log_to_file(file_path, unet, adapter, length, t, r, model_path, acc, acc_low, acc_high,save_path):
    with open(file_path, 'a') as f:
        f.write(f'\nunet: {unet}\n')
        f.write(f'ft_id : {adapter}\n')
        f.write(f'length : {length}\n')
        f.write(f'纠错{t}位，冗余需要{r}位\n')
        f.write(f'model : {model_path}\n')
        f.write(f'bit acc percentage: {acc * 100:.2f}%\n')
        f.write(f'acc_low : {acc_low*100:.2f}%\n')
        f.write(f'acc_high: {acc_high*100:.2f}%\n')
        f.write(f'save_path: {save_path}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--dataset', default='Gustavosta-Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='stable-diffusion-v1-5')
    parser.add_argument('--latent_shape',default=(4,64,64))
    parser.add_argument('--ft_id',default=None,type=str)
    parser.add_argument('--adapter_id',default=None,type=str)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--secret_length', default=48, type=int)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--guidancescale', default=7.5, type=float)
    parser.add_argument('--guidancescale_fanyan', default=1.0, type=float)
    parser.add_argument('--reverse_inference_steps', default=50, type=int)
    parser.add_argument('--place_mode',default='PLACE_SEQUENTIAL')
    parser.add_argument('--threshold',default=1e-6,type=float)
    parser.add_argument('--model', default='inn_model.pth', type=str)
    parser.add_argument('--brightness', default=None, type=float)
    parser.add_argument('--noise', default=None, type=float)
    parser.add_argument('--contrast', default=None, type=float)
    parser.add_argument('--hue', default=None, type=float)
    parser.add_argument('--blur', default=None, type=int)
    parser.add_argument('--jpegcompression', default=None, type=int)
    parser.add_argument('--resize', default=None, type=float)
    parser.add_argument('--save',type=str,default=None)
    parser.add_argument('--margin',type=float,default=1.0)
    parser.add_argument('--total_size',type=int,required=True)
    parser.add_argument('--ecc_backups',type=int,required=True)
    parser.add_argument('--data_backups',type=int,required=True)

    args =parser.parse_known_args()[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_printoptions(sci_mode=False,profile='full')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    maxlength=250
    
    # dataset
    dataset, prompt_key = get_dataset(args)
    dataset=promptdataset(dataset,prompt_key)
    
    # Reed-Solomon
    epsilon = 1e-4
    p_error = 0.1 if args.secret_length<=4096 else 0.04
    best_rs_paras = get_rs_paras(args.secret_length,p_error=p_error,m=8,epsilon=epsilon)
    m, n_rs, k_rs, r_symbols, t_rs, block_size, _, _, ecc_backups = best_rs_paras

    ecc_backups= args.ecc_backups
    data_backups= args.data_backups

    # 如果有空间做纠错冗余，构造域和 RS 对象
    ecc_flag = False
    if n_rs is not None and k_rs is not None:
        GF = galois.GF(2**m)
        rs = galois.ReedSolomon(n_rs, k_rs, field=GF)
        ecc_flag = True


    # save path 
    parent_sp = f'eva_attack_RS/{args.total_size}_{args.secret_length}-{ecc_backups}e-{data_backups}d'
    if not os.path.exists(parent_sp):
        os.mkdir(parent_sp)
    sp=os.path.join(parent_sp,args.save)
    if not os.path.exists(sp):
        os.mkdir(sp)
    


    #model
    if(args.ft_id is not None):
        unet =UNet2DConditionModel.from_pretrained(args.ft_id, subfolder="unet", torch_dtype=torch.float16)
        scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder='scheduler')
        pipe = StableDiffusionPipeline.from_pretrained(
                args.model_path,
                scheduler=scheduler,
                torch_dtype=torch.float16,
                revision='fp16',
                unet=unet,
                )
    if args.adapter_id is not None:
        pipe = StableDiffusionPipeline.from_pretrained(args.model_path,
                                                torch_dtype=torch.float16
                                                )
        pipe.load_lora_weights(args.adapter_id)
        pipe.fuse_lora()

    if args.ft_id is None and args.adapter_id is None:
        scheduler = DDIMScheduler.from_pretrained(args.model_path, subfolder='scheduler')
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float32,
            # revision='fp16',
        )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    inn =newWatermark(args.latent_shape).inn.to(device)
    if args.model !=None:
        inn.load_state_dict(torch.load(args.model))
    inn.eval()

    ACC = []
    img_list = []
    img_list_attacked = []
    for i in tqdm(range(100)):

        if ecc_flag:
            # 这里返回的parity_bits是做测试的，还原信息是用从inverse的latent中提取的parity
            # 这里的embed函数是rs_2，是带有备份原始数据的
            print(f'有空间去容纳至少一份的纠错码')
            secret, positions, inn_input_latent, parity_len, parity_bits = embed_secret_in_latent_rs_2(args.secret_length,
                                                                args.total_size,
                                                                best_rs_paras,
                                                                GF,
                                                                rs,
                                                                ecc_backups,
                                                                data_backups,
                                                                args.latent_shape,
                                                                args.place_mode,
                                                                args.margin)
        
        else:
            # 没空间去做纠错了
            secret, positions, inn_input_latent, parity_len, parity_bits = embed_secret_in_latent_rs_2(args.secret_length,
                                                                args.total_size,
                                                                best_rs_paras,
                                                                None,
                                                                None,
                                                                0, # ecc_backups =0 
                                                                data_backups,
                                                                args.latent_shape,
                                                                args.place_mode,
                                                                args.margin)
            
        secret = torch.tensor(secret).to(device)
        inn_input_latent = torch.tensor(inn_input_latent).to(device).unsqueeze(0)

        z, log_jac_det = inn(inn_input_latent)

        #对latent手动标准化
        mu = z.mean(dim=(0,2,3),keepdim=True)
        std = z.std(dim=(0,2,3),keepdim=True)
        z_norm = (z-mu)/(std+1e-6)


        # vis_inn_input = visualize_latent(inn_input_latent,bin_wid=0.1,latent_shape=args.latent_shape)
        # vis_inn_input.savefig(f'{sp}/inn_input_distribution.png')
        # vis_plt = visualize_latent(z,bin_wid=0.1,latent_shape=args.latent_shape)
        # vis_plt.savefig(f'{sp}/latent_distribution.png')
        # vis_plt_znorm = visualize_latent(z_norm,bin_wid=0.1,latent_shape=args.latent_shape)
        # vis_plt_znorm.savefig(f'{sp}/latent_norm_distribution.png')

        prompt=dataset[-i-1][0:maxlength]
        print(f"current prompt: {prompt}")

        img1= pipe(prompt=prompt,num_inference_steps=args.num_inference_steps,\
        latents=z_norm,guidance_scale=args.guidancescale).images[0]
        img_list.append(img1)

        img1_attacked = img1
        if args.brightness != None:
            img1_attacked = transforms.ColorJitter(brightness=args.brightness)(img1)
        if args.noise != None:
            img1_attacked = np.array(img1, dtype=np.uint8)
            g_noise = np.random.randn(*img1_attacked.shape).astype(np.uint8) * args.noise
            noisy_array = np.clip(img1_attacked.astype(np.float32) + g_noise, 0, 255).astype(np.uint8)
            img1_attacked = Image.fromarray(noisy_array)
        if args.contrast != None:
            enhancer = ImageEnhance.Contrast(img1)
            factor = args.contrast
            img1_attacked = enhancer.enhance(factor)
        if args.hue != None:
            enhancer = ImageEnhance.Color(img1)
            factor = args.hue
            img1_attacked = enhancer.enhance(factor)
        if args.jpegcompression != None:
            img1_attacked = compress_jpeg_to_pil(img1, args.jpegcompression)
        if args.blur != None:
            img1_attacked =Image.fromarray(cv2.GaussianBlur(np.array(img1),(args.blur,args.blur), 1))
        if args.resize != None:
            img1_attacked = img1.resize((int(args.image_length*args.resize), int(args.image_length*args.resize)), PIL.Image.BICUBIC)
        img_list_attacked.append(img1_attacked)
        
        if len(img_list)==5:
            img_stitched = stitch_images(img_list)
            img_stitched.save(f'{sp}/{i-4}-{i}.png')
            img_list.clear()
        if len(img_list_attacked)==5:
            img_stitched = stitch_images(img_list_attacked)
            img_stitched.save(f'{sp}/{i-4}-{i}_attacked.png')
            img_list_attacked.clear()
        reverse_latents=reverse(img1_attacked,pipe,args).float()


        decoded, _ = inn(reverse_latents, rev=True)
        decoded = decoded.flatten()[positions]
        decoded_threshold = (decoded >= 0).int().cpu().numpy()
        # 提取数据位
        data = decoded_threshold[:args.secret_length]
        # 提取数据位
        data = decoded_threshold[:args.secret_length*data_backups]
        data = (data.reshape(-1,args.secret_length)).mean(axis=0)
        data = (data>0.5).astype(int)
        # 提取纠错码
        if ecc_flag:
            len_data = args.secret_length*data_backups
            parity = decoded_threshold[len_data: len_data + parity_len* ecc_backups]
            matrix = parity.reshape(-1, parity_len)   # shape = (复制次数, length)
            col_mean = matrix.mean(axis=0) 
            parity = (col_mean>0.5).astype(int)
            print(f'ecc_backups ={ecc_backups}')
            print(f'纠错码正确率 = {np.mean(parity==parity_bits)}')
            # 纠错
            # parity还原成FieldArray
            parity_blocks = bitarray_to_parity_blocks_8bit(parity,best_rs_paras,GF)
            corrected = decode_rs_blocks(data, parity_blocks, best_rs_paras, GF, rs)
            # 计算使用ECC的ACC
            is_similar = torch.abs(torch.tensor(corrected[:args.secret_length]).to(device) - secret) < args.threshold
            correct_bits = torch.sum(is_similar).item()    
            print(f'>>> correct_bits = {correct_bits}')
            acc = correct_bits/args.secret_length
            ACC.append(acc)
            
            # 计算不使用ECC的ACC
            is_similar_noEcc = torch.sum(torch.tensor(data).to(device)==secret).item()
            acc_noEcc = is_similar_noEcc/args.secret_length

            print('>>> 使用ECC的acc = {:.6f}'.format(acc)+'不使用ECC的acc = {:.6f}'.format(acc_noEcc))

        else:
            # 计算不使用ECC的ACC
            is_similar_noEcc = torch.sum(torch.tensor(data).to(device)==secret).item()
            acc_noEcc = is_similar_noEcc/args.secret_length
            ACC.append(acc_noEcc)                    
            print(f'>>> 不使用ECC的ACC={acc_noEcc}')        
        
        
    acc = sum(ACC)/len(ACC)
    ACC_sorted = sorted(ACC)
    if ecc_flag:
        log_to_file('eva_log_paper.txt',args.ft_id, args.adapter_id,args.secret_length,t_rs*8, parity_len, args.model,acc,ACC_sorted[0],ACC_sorted[-1],sp)
    else:
        log_to_file('eva_log_paper.txt',args.ft_id, args.adapter_id,args.secret_length,0, 0, args.model,acc,ACC_sorted[0],ACC_sorted[-1],sp)


