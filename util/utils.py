import argparse
import os
import sys
from diffusers import DPMSolverMultistepScheduler,DPMSolverMultistepInverseScheduler
import torch
import copy
import numpy as np 
import random 
import matplotlib.pyplot as plt
from datasets import load_dataset
import json
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter,ImageEnhance
import PIL 
import cv2
from io import BytesIO
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch.nn.functional as F
from .ecc_utils import *


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)

def get_dataset(args):
    if 'laion' in args.dataset:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset:
        with open('../fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    elif 'Gustavosta/Stable-Diffusion-Prompts' in args.dataset:
        dataset = load_dataset('../Stable-Diffusion-Prompts/data')['test']
        prompt_key = 'Prompt'
    else:
        dataset = load_dataset(args.dataset)['test']
        prompt_key = 'Prompt'
    return dataset, prompt_key


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0

@torch.inference_mode()
def get_image_latents(pipe, image, sample=True, rng_generator=None):
        encoding_dist = pipe.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        # latents = encoding * 0.13025
        latents = encoding* pipe.vae.config.scaling_factor
        return latents

def get_random_latents(pipe,args,latents=None,generator=None,batch_size=1):
        height = args.image_length
        width = args.image_length
        device = pipe._execution_device
        num_channels_latents = pipe.unet.config.in_channels
        latents = pipe.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            pipe.text_encoder.dtype,
            device,
            generator,
            latents,
        )
        return latents

class promptdataset(Dataset):
    def __init__(self, data_list , prompt_key):
        self.data = data_list
        self.prompt_key= prompt_key
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index][self.prompt_key]
    
def reverse(image,pipe,args,prompt=''):
    curr_scheduler = pipe.scheduler
    # pipe.scheduler =DPMSolverMultistepInverseScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe.scheduler =DDIMInverseScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe.vae.to(torch.float32)
    img = transform_img(image,args.image_length).unsqueeze(0).to(pipe.vae.dtype).to(pipe.device)
    image_latents=get_image_latents(pipe, img, sample=False)
    image_latents=image_latents.to(pipe.unet.dtype)
    inverted_latents = pipe(prompt=prompt, latents=image_latents, num_inference_steps=args.reverse_inference_steps, output_type="latent",guidance_scale=args.guidancescale)
    inverted_latents = inverted_latents.images
    pipe.scheduler = curr_scheduler
    pipe.vae.to(pipe.unet.dtype) 
    return inverted_latents

def compress_jpeg_to_pil(img, quality):
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG', quality=quality)
    output_buffer.seek(0)
    return Image.open(output_buffer)


def adversarial_samples(img,batch,device,X,args):
        img_tmp=img.copy()
        for noise in [0.1,0.3,0.4,0.5]:
                for t,image in enumerate(img_tmp):
                    img1 = np.array(image, dtype=np.uint16)
                    g_noise = np.random.randn(*img1.shape).astype(np.uint8)*noise 
                    noisy_array = np.clip(img1.astype(np.uint16) + g_noise, 0, 255).astype(np.uint8)
                    img1 = Image.fromarray(noisy_array)
                    img.append(img1)
                    batch=torch.cat((batch,X[t]),dim=0).to(device)
        for compress_scale in [50,70,90]:
            for t,image in enumerate(img_tmp):
                img.append(compress_jpeg_to_pil(image, compress_scale))
                batch=torch.cat((batch,X[t]),dim=0).to(device)
        for resizescale in [0.3,0.6,0.9]:
            for t,image in enumerate(img_tmp):
                img.append(image.resize((int(args.image_length * resizescale), int(args.image_length * resizescale)), PIL.Image.BICUBIC))
                batch=torch.cat((batch,X[t]),dim=0).to(device)
        for kernelsize in [3,5,7]:
            for t,image in enumerate(img_tmp):
                blurred_array=cv2.GaussianBlur(np.array(image), (kernelsize, kernelsize), 0)
                img.append(Image.fromarray(blurred_array))
                batch=torch.cat((batch,X[t]),dim=0).to(device)
        for brightness in [1,2]:
            for t,image in enumerate(img_tmp):
                img.append(transforms.ColorJitter(brightness=brightness)(image))
                batch=torch.cat((batch,X[t]),dim=0).to(device)
        for contrast in [1,2]:
            for t,image in enumerate(img_tmp):
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)
                img.append(image)
                batch=torch.cat((batch,X[t]),dim=0).to(device)
        return img,batch


def generate_numbers_from_secret(secret,margin):
    """
    根据输入的 secret 数组生成对应的正数和负数数组。
    
    参数:
        secret (np.ndarray): 包含 0 和 1 的 numpy 数组。
        
    返回:
        np.ndarray: 对应的正数和负数数组。
    """
    # 确保输入是一个 numpy 数组
    secret = np.array(secret)
    
    # 生成标准正态分布的随机数
    random_numbers = np.random.randn(*secret.shape)
    result = np.where(secret == 0, -np.abs(random_numbers)-margin, np.abs(random_numbers)+margin)
    # result = np.where(secret == 0, -np.abs(random_numbers), np.abs(random_numbers))
    
    return result


def embed_secret_in_latent(secret_length, latent_shape=(4, 64, 64), place_mode=None, margin=1.0):
    """
    将水印信息嵌入到一个 latent 空间中，其余位置填充随机噪声。
    
    参数：
      secret_length: 信息长度
      latent_shape: latent 张量的形状，默认为 (4, 64, 64)。
      place_mode:固定 secret 位置的策略
      positions: 一个整数数组，表示在扁平化的 latent 中秘密内容的固定位置。如果为 None，则默认均匀选取。
    返回值：
        secret: 水印信息
        positions: 在固定latent shape和place mode下水印信息在latent中的位置
        latent:输入给INN训练的latent,目的是让INN可以做到根据z（即经过INN后的latent）完全逆向来提取水印信息
        secret和latent都是np array
    """ 
    secret = np.random.randint(0,2,size=secret_length)
    secret_insert =generate_numbers_from_secret(secret,margin)
    # print(f'secret_insert = {secret_insert[:20]}')
    # print(f'secret_insert shape = {secret_insert.shape}')
    # 计算 latent 总的元素数
    total_positions = np.prod(latent_shape)

    # 如果没有指定固定位置，则均匀地选择 secret_length 个位置
    if place_mode =='PLACE_LINSPACE':
        positions = np.linspace(0, total_positions - 1, num=secret_length, dtype=int)
    elif place_mode == 'PLACE_SEQUENTIAL':
        positions = np.arange(0,secret_length)
    elif place_mode == 'PLACE_CHANNEL_SEQ':
        t = latent_shape[1]*latent_shape[2]
        first_channel_positions = np.arange(0,secret_length // latent_shape[0])
        positions = np.concatenate((first_channel_positions, first_channel_positions + t), axis=0)
        for i in range(latent_shape[0]-2):
            positions = np.concatenate((positions, first_channel_positions + t*(i+2)), axis=0)
    else:
        raise ValueError("Place mode is unknown. You should CHOOSE FROM")
    # print(f'positions:\n{positions}')

    # 生成一个含随机噪声的 latent 张量（这里使用标准正态分布，可以根据需求调整）
    latent = np.random.randn(*latent_shape).astype(np.float32)
    
    # 将 latent 展平成一维数组便于替换secret
    latent_flat = latent.flatten()
    
    # 检查 secret 的长度是否超出指定位置数
    if secret_length > len(positions):
        raise ValueError("secret 的长度超过了所提供的固定位置数量。")
    
    # 在固定位置插入 secret 内容
    latent_flat[positions] = secret_insert
    
    # 重新 reshape 成指定的 latent 形状
    latent = latent_flat.reshape(latent_shape)
    
    return secret, positions, latent


def embed_secret_with_backup(length,latent_shape,place_mode,margin,total_info_size,backup_counts):
    """
    对水印信息进行备份，不使用ECC，剩下的位置随机填充到total_info_size
    """
    secret = np.random.randint(0,2,size=length)
    # 拷贝原始数据
    secret_backuped = np.resize(secret, backup_counts*length)   
    # 剩下位置进行随机填充
    s = total_info_size-backup_counts*length
    padding = np.random.randint(0,2,size=s)
    info_insert = np.concatenate((secret_backuped, padding))
    # 生成插入的信息
    secret_insert =generate_numbers_from_secret(info_insert,margin)
    # print(f'水印长度={length}，拷贝后数据长度={len(secret_backuped)}，填充长度={len(padding)},插入信息长度={len(info_insert)}')
    # sys.exit()

    # 计算 latent 总的元素数
    total_positions = np.prod(latent_shape)

    # 如果没有指定固定位置，则均匀地选择 total_pos 个位置
    if place_mode is None:
        positions = np.linspace(0, total_positions - 1, num=total_info_size, dtype=int)
    elif place_mode == 'PLACE_SEQUENTIAL':
        positions = np.arange(0,total_info_size)  
    elif place_mode == 'PLACE_CHANNEL_SEQ':
        t = latent_shape[1]*latent_shape[2]
        first_channel_positions = np.arange(0,total_info_size // latent_shape[0])
        positions = np.concatenate((first_channel_positions, first_channel_positions + t), axis=0)
        for i in range(latent_shape[0]-2):
            positions = np.concatenate((positions, first_channel_positions + t*(i+2)), axis=0)
    else:
        raise ValueError("Place mode is unknown. ")
    
    # 生成一个含随机噪声的 latent 张量（这里使用标准正态分布，可以根据需求调整）
    latent = np.random.randn(*latent_shape).astype(np.float32)
    
    # 将 latent 展平成一维数组便于替换secret
    latent_flat = latent.flatten()
    
    # 在固定位置插入 secret_insert 内容
    latent_flat[positions] = secret_insert
    
    # 重新 reshape 成指定的 latent 形状
    latent = latent_flat.reshape(latent_shape)
    
    return secret, positions, latent

def embed_secret_in_latent_ecc(secret_length, total_info_size, latent_shape=(4, 64, 64), place_mode=None, margin=1.0):
    """
    将水印信息嵌入到一个 latent 空间中，剩余位置复制最多份的冗余码，其余位置填充随机噪声。
    
    参数：
      secret_length: 信息长度
      latent_shape: latent 张量的形状，默认为 (4, 64, 64)。
      place_mode:固定 secret 位置的策略
      positions: 一个整数数组，表示在扁平化的 latent 中秘密内容的固定位置。。
      total_info_size:训练时候嵌入信息的size
    返回值：
        secret: 水印信息,后续用于evaluation
        positions: 在固定latent shape和place mode下水印信息在latent中的位置
        latent:输入给INN训练的latent
        parity_len:用于evaluation时提取冗余位

    """ 
    secret = np.random.randint(0,2,size=secret_length)
    # 根据secret计算ECC码
    secret_str = ''.join(map(str, secret))
    data_str, parity_str = encode_bitstring_separate(secret_str, block_size=16)
    # print(f'>>> secret_length ={(secret_length)}')
    # print(f'>>> secret_str len ={len(secret_str)}')
    # print(f'>>> data_str len ={len(data_str)},parity_str len ={len(parity_str)}')

    # 复制冗余位
    parity_len = len(parity_str)
    t = (total_info_size - secret_length) // parity_len
    parity_backup = parity_str * t
    # 填充剩余的位
    s = total_info_size-secret_length - parity_len*t
    padding = np.random.randint(0,2,size=s)
    padding_str = str(padding)
    padding_str = ''.join([c for c in padding_str if c in ('0', '1')])
    ecc_str = data_str+parity_backup  if s==0 else data_str+parity_backup+padding_str
    
    # print(f's={s},t={t}')
    # print(f'parity_len={parity_len}')
    # print(f'ecc_str={len(ecc_str)}')
    # print(f'data_str={len(data_str)}')
    # print(f'parity_backup={len(parity_backup)}')
    # sys.exit()
    ecc_np = np.array([int(x) for x in ecc_str])

    ecc_insert =generate_numbers_from_secret(ecc_np,margin)

    # 计算 latent 总的元素数
    total_positions = np.prod(latent_shape)
    info_length = len(ecc_insert)
    # print(f'>>> info_length ={info_length}')

    if place_mode =='PLACE_LINSPACE':
        positions = np.linspace(0, total_positions - 1, num=info_length, dtype=int)
    elif place_mode == 'PLACE_SEQUENTIAL':
        positions = np.arange(0,info_length)
    elif place_mode == 'PLACE_CHANNEL_SEQ':
        t = latent_shape[1]*latent_shape[2]
        first_channel_positions = np.arange(0,info_length // latent_shape[0])
        positions = np.concatenate((first_channel_positions, first_channel_positions + t), axis=0)
        for i in range(latent_shape[0]-2):
            positions = np.concatenate((positions, first_channel_positions + t*(i+2)), axis=0)
    else:
        raise ValueError("Place mode is unknown.")

    # 生成一个含随机噪声的 latent 张量（这里使用标准正态分布，可以根据需求调整）
    latent = np.random.randn(*latent_shape).astype(np.float32)
    
    # 将 latent 展平成一维数组便于替换secret
    latent_flat = latent.flatten()
    
    # 检查长度是否超出指定位置数
    if info_length > len(positions):
        raise ValueError("secret 的长度超过了所提供的固定位置数量。")
    
    # 在固定位置插入 secret 内容
    latent_flat[positions] = ecc_insert
    
    # 重新 reshape 成指定的 latent 形状
    latent = latent_flat.reshape(latent_shape)
    
    return secret, positions, latent, parity_len,parity_str

def embed_secret_in_latent_bch(secret_length, total_info_size, bch, best_bch_paras, backup_r, latent_shape=(4, 64, 64), place_mode=None, margin=1.0):
    """
    将水印信息嵌入到一个 latent 空间中，复制backup_r份冗余位，其余位置填充随机噪声。
    
    参数：
      secret_length: 信息长度
      latent_shape: latent 张量的形状，默认为 (4, 64, 64)。
      place_mode:固定 secret 位置的策略
      positions: 一个整数数组，表示在扁平化的 latent 中秘密内容的固定位置。。
      total_info_size:训练时候嵌入信息的size
      bch:bch多项式
    返回值：
        secret: 水印信息,后续用于evaluation
        positions: 在固定latent shape和place mode下水印信息在latent中的位置
        latent:输入给INN训练的latent
        parity_len:用于evaluation时提取冗余位

    """ 
    secret = np.random.randint(0,2,size=secret_length)
    # 根据secret计算BCH
    data_padded, parity_bits = encode_bch_blocks(secret,bch,best_bch_paras)

    # 复制冗余位
    parity_len = len(parity_bits)
    backup_counts = (total_info_size - secret_length) // parity_len
    backup_counts = min(backup_counts,backup_r)
    parity_backup = np.tile(parity_bits, backup_counts)
    # 填充剩余的位
    s = total_info_size-len(data_padded) - parity_len*backup_counts
    padding = np.random.randint(0,2,size=s)
    # 最后嵌入的信息，扩充64倍数的原始信息+复制后的冗余+填充
    ecc_np = np.concatenate((data_padded, parity_backup))
    ecc_np = np.concatenate((ecc_np, padding))

    ecc_insert =generate_numbers_from_secret(ecc_np,margin)

    # 计算 latent 总的元素数
    total_positions = np.prod(latent_shape)
    info_length = len(ecc_insert)
    # print(f'>>> info_length ={info_length}')

    if place_mode =='PLACE_LINSPACE':
        positions = np.linspace(0, total_positions - 1, num=info_length, dtype=int)
    elif place_mode == 'PLACE_SEQUENTIAL':
        positions = np.arange(0,info_length)
    elif place_mode == 'PLACE_CHANNEL_SEQ':
        a = latent_shape[1]*latent_shape[2]
        first_channel_positions = np.arange(0,info_length // latent_shape[0])
        positions = np.concatenate((first_channel_positions, first_channel_positions + a), axis=0)
        for i in range(latent_shape[0]-2):
            positions = np.concatenate((positions, first_channel_positions + a*(i+2)), axis=0)
    else:
        raise ValueError("Place mode is unknown.")

    # 生成一个含随机噪声的 latent 张量（这里使用标准正态分布，可以根据需求调整）
    latent = np.random.randn(*latent_shape).astype(np.float32)
    
    # 将 latent 展平成一维数组便于替换secret
    latent_flat = latent.flatten()
    
    # 检查长度是否超出指定位置数
    if info_length > len(positions):
        raise ValueError("secret 的长度超过了所提供的固定位置数量。")
    

    # print(f'positions len = {len(positions)}')
    # print(f'secret len={len(secret)},r len={parity_len},backup_r len = {len(parity_backup)},padding len={len(padding)}, s+br+p ={len(data_padded)+len(parity_backup)+len(padding)}')
    # sys.exit()


    # 在固定位置插入 secret 内容
    latent_flat[positions] = ecc_insert
    
    # 重新 reshape 成指定的 latent 形状
    latent = latent_flat.reshape(latent_shape)
    
    return secret, positions, latent, parity_len, parity_bits

def embed_secret_in_latent_rs(secret_length, total_info_size, paras, GF, rs, backup_r, latent_shape=(4, 64, 64), place_mode=None, margin=1.0):
    """
    将水印信息嵌入到一个 latent 空间中，复制backup_r份冗余位，其余位置填充随机噪声。

    """ 
    secret = np.random.randint(0,2,size=secret_length)
    if backup_r>0:
        # 根据secret计算RS冗余
        parity_blocks = encode_rs_bitstring(secret,paras,GF,rs)
        # 转成一维 uint8 ndarray
        parity_flat_np = parity_blocks_to_bitarray_8bit(parity_blocks)
        
        # 复制冗余位
        parity_len = len(parity_flat_np)
        backup_counts = (total_info_size - secret_length) // parity_len
        backup_counts = min(backup_counts,backup_r)
        parity_backup = np.tile(parity_flat_np, backup_counts)
        # 填充剩余的位
        s = total_info_size-len(secret) - len(parity_backup)
        padding = np.random.randint(0,2,size=s)
        # 最后嵌入的信息，原始信息+复制后的冗余+填充
        ecc_np = np.concatenate((secret, parity_backup))
        ecc_np = np.concatenate((ecc_np, padding))
    else:
        parity_len=0
        parity_flat_np=None
        s = total_info_size-len(secret)
        padding = np.random.randint(0,2,size=s)
        ecc_np = np.concatenate((secret,padding))

    ecc_insert =generate_numbers_from_secret(ecc_np,margin)

    # 计算 latent 总的元素数
    total_positions = np.prod(latent_shape)
    info_length = len(ecc_insert)
    # print(f'>>> info_length ={info_length}')

    if place_mode =='PLACE_LINSPACE':
        positions = np.linspace(0, total_positions - 1, num=info_length, dtype=int)
    elif place_mode == 'PLACE_SEQUENTIAL':
        positions = np.arange(0,info_length)
    elif place_mode == 'PLACE_CHANNEL_SEQ':
        a = latent_shape[1]*latent_shape[2]
        first_channel_positions = np.arange(0,info_length // latent_shape[0])
        positions = np.concatenate((first_channel_positions, first_channel_positions + a), axis=0)
        for i in range(latent_shape[0]-2):
            positions = np.concatenate((positions, first_channel_positions + a*(i+2)), axis=0)
    else:
        raise ValueError("Place mode is unknown.")

    # 生成一个含随机噪声的 latent 张量（这里使用标准正态分布，可以根据需求调整）
    latent = np.random.randn(*latent_shape).astype(np.float32)
    
    # 将 latent 展平成一维数组便于替换secret
    latent_flat = latent.flatten()
    
    # 检查长度是否超出指定位置数
    if info_length > len(positions):
        raise ValueError("secret 的长度超过了所提供的固定位置数量。")
    

    # print(f'positions len = {len(positions)}')
    # print(f'secret len={len(secret)},r len={parity_len},backup_r len = {len(parity_backup)},padding len={len(padding)}, s+br+p ={len(data_padded)+len(parity_backup)+len(padding)}')
    # sys.exit()


    # 在固定位置插入 secret 内容
    latent_flat[positions] = ecc_insert
    
    # 重新 reshape 成指定的 latent 形状
    latent = latent_flat.reshape(latent_shape)
    
    return secret, positions, latent, parity_len, parity_flat_np


def embed_secret_in_latent_rs_2(secret_length, total_info_size, paras, GF, rs, backup_r, backup_d, latent_shape=(4, 64, 64), place_mode=None, margin=1.0):
    """
    将水印信息嵌入到一个 latent 空间中，复制backup_d份数据，复制backup_r份冗余位，其余位置填充随机噪声。

    """ 
    secret = np.random.randint(0,2,size=secret_length)
    secret_backuped = np.tile(secret,backup_d)
    if backup_r>0:
        # 根据secret计算RS冗余
        parity_blocks = encode_rs_bitstring(secret,paras,GF,rs)
        # 转成一维 uint8 ndarray
        parity_flat_np = parity_blocks_to_bitarray_8bit(parity_blocks)
        
        # 复制冗余位
        parity_len = len(parity_flat_np)
        backup_counts = (total_info_size - secret_length) // parity_len
        backup_counts = min(backup_counts,backup_r)
        parity_backup = np.tile(parity_flat_np, backup_counts)

        # check
        # print(f'len of secret_backuped = {len(secret_backuped)}')
        # print(f'len of parity_backup = {len(parity_backup)}')
        # sys.exit()

        # 填充剩余的位
        s = total_info_size-len(secret_backuped) - len(parity_backup)
        padding = np.random.randint(0,2,size=s)
        # 最后嵌入的信息，原始信息+复制后的冗余+填充
        ecc_np = np.concatenate((secret_backuped, parity_backup))
        ecc_np = np.concatenate((ecc_np, padding))
    else:
        parity_len=0
        parity_flat_np=None
        s = total_info_size-len(secret_backuped)
        padding = np.random.randint(0,2,size=s)
        ecc_np = np.concatenate((secret_backuped,padding))


    ecc_insert =generate_numbers_from_secret(ecc_np,margin)

    # 计算 latent 总的元素数
    total_positions = np.prod(latent_shape)
    info_length = len(ecc_insert)
    # print(f'>>> info_length ={info_length}')

    if place_mode =='PLACE_LINSPACE':
        positions = np.linspace(0, total_positions - 1, num=info_length, dtype=int)
    elif place_mode == 'PLACE_SEQUENTIAL':
        positions = np.arange(0,info_length)
    elif place_mode == 'PLACE_CHANNEL_SEQ':
        a = latent_shape[1]*latent_shape[2]
        first_channel_positions = np.arange(0,info_length // latent_shape[0])
        positions = np.concatenate((first_channel_positions, first_channel_positions + a), axis=0)
        for i in range(latent_shape[0]-2):
            positions = np.concatenate((positions, first_channel_positions + a*(i+2)), axis=0)
    else:
        raise ValueError("Place mode is unknown.")

    # 生成一个含随机噪声的 latent 张量（这里使用标准正态分布，可以根据需求调整）
    latent = np.random.randn(*latent_shape).astype(np.float32)
    
    # 将 latent 展平成一维数组便于替换secret
    latent_flat = latent.flatten()
    
    # 检查长度是否超出指定位置数
    if info_length > len(positions):
        raise ValueError("secret 的长度超过了所提供的固定位置数量。")
    

    # print(f'positions len = {len(positions)}')
    # print(f'secret len={len(secret)},r len={parity_len},backup_r len = {len(parity_backup)},padding len={len(padding)}, s+br+p ={len(data_padded)+len(parity_backup)+len(padding)}')
    # sys.exit()


    # 在固定位置插入 secret 内容
    latent_flat[positions] = ecc_insert
    
    # 重新 reshape 成指定的 latent 形状
    latent = latent_flat.reshape(latent_shape)
    
    return secret, positions, latent, parity_len, parity_flat_np


    


# def visualize_latent(latent, bin_wid, latent_shape ,bin_min=-3.0, bin_max=3.0):
#     bins = np.arange(bin_min, bin_max + bin_wid, bin_wid)
#     latent_flatten = latent.flatten().cpu()
#     latent_flatten = latent_flatten.detach().numpy()
#     hist, bin_edges = np.histogram(latent_flatten, bins=bins)

#     plt.figure(figsize=(10,6))
#     # 绘制 latent 分布
#     plt.bar(
#         bin_edges[:-1],
#         hist,
#         width=bin_wid,
#         align='edge',
#         alpha=0.7,
#         label='INN latent z',
#         color='blue'
#     )

#     # 绘制标准正态分布
#     z_target = np.random.randn(*(latent_shape)).reshape(-1)
#     hist, bin_edges = np.histogram(z_target, bins=bins)
#     plt.bar(
#         bin_edges[:-1],
#         hist,
#         width=bin_wid,
#         align='edge',
#         alpha=0.7,
#         label='Standard Normal Distribution',
#         color='green'
#     )
#     plt.xlabel('Z Value')
#     plt.ylabel('Count')
#     plt.title('Distribution of INN Latent Variables z')
#     plt.legend()
#     # plt.savefig('latent_distribution.png',dpi=600)
#     return plt

def visualize_latent(latent, bin_wid, latent_shape, auto_range=True):
    # print("\n[输入数据校验]")
    # print("数据类型:", type(latent))
    # if isinstance(latent, torch.Tensor):
    #     latent_np = latent.detach().cpu().numpy().flatten()
    # else:
    #     latent_np = np.array(latent).flatten()
    # print("最小值:", latent_np.min())
    # print("最大值:", latent_np.max())
    # print("均值:", latent_np.mean())
    # print("标准差:", latent_np.std())
    # print(">1的比例:", f"{(latent_np > 1.0).sum() / len(latent_np):.2%}")
    # print("<-1的比例:", f"{(latent_np < -1.0).sum() / len(latent_np):.2%}")
    # 转换数据
    latent_flatten = latent.detach().cpu().numpy().flatten()
    
    # 自动计算分箱范围
    # if auto_range:
    #     data_min = min(latent_flatten.min(), -5.0)
    #     data_max = max(latent_flatten.max(), 5.0)
    # else:
    #     data_min, data_max = -5.0, 5.0  # 覆盖绝大多数情况
    
    data_min, data_max = -5.0, 5.0  # 覆盖绝大多数情况
    bins = np.arange(data_min, data_max + bin_wid, bin_wid)

    plt.figure(figsize=(10,6))
    
    # 绘制真实 latent 分布（红色半透明）
    plt.hist(latent_flatten, bins=bins, alpha=0.5, 
            label='Actual Latent', color='red', density=True)
    
    # 绘制理论标准正态分布（更大样本量）
    z_target = np.random.randn(10 * np.prod(latent_shape))  # 10倍样本量
    plt.hist(z_target, bins=bins, alpha=0.5, 
            label='Theoretical Normal', color='blue', density=True)
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Latent Distribution Comparison')
    plt.legend()
    return plt

def record_training_data(epoch, loss_mse, log_jac_det, loss_adv, loss_inn, loss_d, accuracy, pz, secret=None, decoded=None, file_path='training_data.json'):
    secret_list = None
    decoded_list = None
    if secret is not None and decoded is not None:
        secret_list = secret.tolist()
        decoded_list = decoded.tolist()
    training_data = {
        'epoch': epoch,
        'loss_mse': loss_mse.item(),
        'log_jac_det': log_jac_det.item(), 
        'loss_adv':loss_adv.item(),
        'loss_inn': loss_inn.item(), 
        'loss_d':loss_d.item(),
        'accuracy': accuracy,
        'pz':pz.item(),
        'secret': secret_list,
        'decoded': decoded_list
    }

    # 如果文件不存在或为空，初始化为一个空列表
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        with open(file_path, 'w') as f:
            json.dump([training_data], f, indent=4)  # 初始化为包含第一个数据点的列表
    else:
        # 文件存在且非空，读取现有内容并追加新数据
        try:
            with open(file_path, 'r+') as f:
                data = json.load(f)  # 读取现有数据
                data.append(training_data)  # 添加新的数据
                f.seek(0)  # 将文件指针回到文件开头
                json.dump(data, f, indent=4)  # 写回更新后的数据
        except json.JSONDecodeError:
            # 如果文件中内容有问题（例如文件格式不正确），初始化为一个有效的列表
            with open(file_path, 'w') as f:
                json.dump([training_data], f, indent=4)  # 重新初始化文件

def draw_loss_curve(epoch, inn_loss_his, disc_loss_his, mse_loss_his, adv_loss_his, args, file_path):
    """
    绘制训练过程中 loss 曲线，每 5 个 epoch 记录一次 loss 情况并绘制。

    参数：
    - epoch: 当前的 epoch 数（整型）。
    - loss_his: 一个列表，记录了每 5 个 epoch 的 loss 值。
    - args: 包含训练参数的对象，此处使用其中的 secret_length 作为示例。
    """
    # 每 5 个 epoch 绘制一次
    if (epoch+1) % 5 == 0:
        # 构造 x 轴数据：假设 loss_his 中记录的 loss 对应的 epoch 分别为 5, 10, ..., epoch
        epochs = list(range(5, epoch + 2, 5))
        inn_loss_his = [loss.detach().cpu().numpy() if isinstance(loss,torch.Tensor) else loss for loss in inn_loss_his]
        disc_loss_his = [loss.detach().cpu().numpy() if isinstance(loss,torch.Tensor) else loss for loss in disc_loss_his]
        mse_loss_his = [loss.detach().cpu().numpy() if isinstance(loss,torch.Tensor) else loss for loss in mse_loss_his]
        adv_loss_his = [loss.detach().cpu().numpy() if isinstance(loss,torch.Tensor) else loss for loss in adv_loss_his]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, inn_loss_his, marker=',', color='blue', label='inn')
        plt.plot(epochs, disc_loss_his, marker=',', color='green', label='disc')
        plt.plot(epochs, mse_loss_his, marker=',', color='red', label='mse')
        plt.plot(epochs, adv_loss_his, marker=',', color='m', label='adv')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f"Training Loss Curve (Secret Length: {args.secret_length})")
        plt.legend()
        plt.grid(True)
        # 保存图像文件（可选）
        plt.savefig(file_path)


# WGAN-GP 判别器损失
def compute_gradient_penalty(D, real_samples, fake_samples):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    alpha = torch.rand(real_samples.size(0),1,1,1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)  # 展平所有空间和通道维度
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def channel_wise_mmd_loss(z_target, z_fake, sigma=1.0):
    """
    输入 z_target 和 z_fake 形状: [batch, channels, height, width]
    输出逐通道的MMD损失均值
    """
    batch_size, channels, h, w = z_target.shape
    total_loss = 0.0
    
    for c in range(channels):
        # 提取当前通道的数据 [batch, h, w]
        target_c = z_target[:, c, :, :].view(batch_size, -1)  # [batch, h*w]
        fake_c = z_fake[:, c, :, :].view(batch_size, -1)      # [batch, h*w]
        
        # 计算当前通道的MMD损失
        def gaussian_kernel(x, y):
            x = x.unsqueeze(1)  # [batch_x, 1, features]
            y = y.unsqueeze(0)  # [1, batch_y, features]
            squared_dist = torch.sum((x - y) ** 2, dim=-1)
            return torch.exp(-squared_dist / (2 * sigma ** 2))
        
        k_target = gaussian_kernel(target_c, target_c).mean()
        k_fake = gaussian_kernel(fake_c, fake_c).mean()
        k_cross = gaussian_kernel(target_c, fake_c).mean()
        loss_c = k_target + k_fake - 2 * k_cross
        
        total_loss += loss_c
    
    # 返回平均损失
    return total_loss / channels


# 假设输入为x，INN输出为z, log_jac_det
def mle_loss(z, log_jac_det):
    """
    输入z的形状应该是(batch_size,4,64,64),batch_size=1,因为这里是每次用一条prompt去训练
    """
    batch_size = z.shape[0]
    D = z[0].numel()  # 4x64x64=16384
    log_prob = -0.5 * torch.sum(z**2, dim=(1,2,3)) - 0.5 * D * np.log(2*np.pi)
    loss = -torch.mean(log_prob + log_jac_det) / D  # 按维度归一化
    return loss

# 整体矩匹配损失
def moment_loss(z):
    """
    输入z的形状应该是(batch_size,4,64,64),batch_size=1,因为这里是每次用一条prompt去训练
    """
    mean_loss = torch.mean(z)**2	#强制均值为0
    var_loss = (torch.var(z)-1.0)**2	#强制方差为1
    return mean_loss+var_loss

# 求各个通道的矩匹配损失平均值
def channel_moment_loss(z):
    """
    输入z的形状应该是(batch_size,4,64,64),batch_size=1,因为这里是每次用一条prompt去训练
    """
    channel_mean = torch.mean(z,dim=(0,2,3))
    channel_var = torch.var(z,dim=(0,2,3))
    mean_loss = torch.mean(channel_mean**2)
    var_loss = torch.mean((channel_var-1.0)**2)

    return mean_loss+var_loss

def mle_loss_video(z, log_jac_det, reduce='mean', per_dim=True, bits_per_dim=False, eps=0.0):
    """
    z: (B, C, F, H, W) —— INN 的输出
    log_jac_det: (B,) 或可广播到 (B,) —— 对应每个样本的 log|det ∂f/∂x|
    reduce: 'mean' | 'sum' | None
    per_dim: True 则按维度归一化为 NLL per-dim
    bits_per_dim: True 则返回 bits/dim（= nats/dim / ln 2）
    eps: 可选的小正数，若你在前面对 z 做了轻微裁剪/噪声
    """
    B = z.shape[0]
    D = z[0].numel()  # = C*F*H*W

    # log p(z) for standard normal
    # 注意把 (C,F,H,W) 展平成一维后按样本求和
    z2_sum = z.reshape(B, -1).pow(2).sum(dim=1)                  # (B,)
    log_const = -0.5 * D * math.log(2.0 * math.pi)
    log_prob_z = -0.5 * z2_sum + log_const                       # (B,)

    # total log-likelihood per sample
    # 形状兼容即可：若 log_jac_det 是标量，会广播；推荐每样本一个值
    log_lik = log_prob_z + log_jac_det                           # (B,)

    # negative log-likelihood
    nll = -log_lik                                                # (B,)

    # 归一化：per-dim 或 bits/dim（常用于可比性）
    if per_dim:
        nll = nll / D
    if bits_per_dim:
        nll = nll / math.log(2.0)

    # 归约
    if reduce == 'mean':
        return nll.mean()
    elif reduce == 'sum':
        return nll.sum()
    else:
        return nll  # (B,)
    
def monitor_output_stats(z, step=None):
    """
    z: (B, C, F, H, W)
    打印每一帧的均值和方差，按通道汇总
    """
    B, C, F, H, W = z.shape

    if step is not None:
        print(f"\n[Step {step}] Output stats:")

    for f in range(F):
        frame_data = z[:, :, f, :, :]   # (B, C, H, W)
        mean = frame_data.mean(dim=(0,2,3))  # (C,)
        var  = frame_data.var(dim=(0,2,3), correction=0)  # (C,)
        # detach 后再转 numpy
        mean_np = mean.detach().cpu().numpy()
        var_np  = var.detach().cpu().numpy()
        print(f"  Frame {f:02d}: mean={mean_np}, var={var_np}")

def channel_moment_loss_video(z):
    # z: (B, C, F, H, W)
    # 对每个 (c,f) 统计，均值/方差在 (H,W) 以及 B 上聚合
    mean = z.mean(dim=(0, 3, 4))                         # (C, F)
    var  = z.var(dim=(0, 3, 4), correction=0)            # (C, F) 用总体方差，避免不必要偏置
    mean_loss = (mean**2).mean()
    var_loss  = ((var - 1.0)**2).mean()
    return mean_loss + var_loss


def ortho_regularization(encoder,info_dim):
    P = encoder.proj.weight  # shape (latent_flat_dim, info_dim)
    return torch.norm(P.t() @ P - torch.eye(info_dim, device=P.device))


def stitch_images(imgs):
    # 检查输入的图像列表是否包含五张图像
    if len(imgs) != 5:
        raise ValueError("输入的图像列表必须包含五张图像。")

    # 定义每张图像的宽度和高度
    img_width, img_height = 512, 512

    # 创建一个新的空白图像，用于拼接
    result_width = img_width * 5
    result_height = img_height 
    result_image = Image.new('RGB', (result_width, result_height))

    # 拼接第一行
    for i in range(5):
        result_image.paste(imgs[i], (i * img_width, 0))

    return result_image


def sign_bce_loss(y_pred, y_true, k=1.0, alpha=0.2):
    """
    y_pred, y_true: (B, …)  – 任意形状
    k      : logit 放大系数，加大决策边界
    alpha  : MSE 与 BCE 的权衡系数
    """

    #  符号 BCE‑with‑logits
    loss_sign = F.binary_cross_entropy_with_logits(k * y_pred, y_true)

    #  幅值 MSE（可选）
    loss_mag  = F.mse_loss(y_pred, y_true)

    return loss_sign + alpha * loss_mag

def margin_mse_loss(decoded, secret_bits, margin=1.5, use_tanh=False):
    """
    decoded     : (B,L) or (...,)   — layer2 的原始输出 (float)
    secret_bits : 同形状             — 0/1 或 bool
    margin      : 正值；建议 1.5~2.5
    use_tanh    : 若 True，先经过 tanh 再放大
    ----------------------------------------------
    目标 y = ±margin，其中 0 -> -margin, 1 -> +margin
    损失 = MSE(decoded, y)
    """
    # 0/1 → -1/+1，再乘 margin
    target = (secret_bits.float() * 2 - 1) * margin

    if use_tanh:                      # 常配合 layer2 的 tanh 激活
        decoded = torch.tanh(decoded) * margin

    return F.mse_loss(decoded, target)


def dist_loss(latent, log_jac_det):
    # 依然保留 MLE，确保 sigma 不会发散
    nll = mle_loss(latent, log_jac_det)                # or -log p(z)
    
    # 用简易 Energy Distance / MMD 压尾巴 & 峭度
    z   = latent.flatten(2)               # (B,C,N)
    # print(f'z shape = {z.shape}')
    z_t = torch.randn_like(z)             # 标准正态
    mmd = ((z.mean(-1) - z_t.mean(-1))**2 +
           (z.var(-1)  - z_t.var(-1)) **2).mean()
    
    # kurt = ((latent**4).mean(dim=[2,3]) - 3).pow(2).mean()  # 峭度
    mu   = latent.mean((2,3))            # (B,C)
    std  = latent.std((2,3))
    ch   = (mu.pow(2) + (std-1).pow(2)).mean()
    return nll + 0.2 * mmd + ch

def log_args_to_file(args: argparse.Namespace, log_file_path: str = "train_log.txt"):
    """
    把 argparse 的参数记录到指定文件
    
    参数:
    - args: 由 argparse 解析得到的参数
    - log_file_path: 日志文件的路径，默认为 "train_log.txt"
    """
    # 提取目录路径
    dir_path = os.path.dirname(log_file_path)
    
    # 若目录不存在，则创建
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # 把参数写入文件
    with open(log_file_path, 'w') as f:
        f.write("===== Training Arguments =====\n")
        for arg_name, arg_value in vars(args).items():
            f.write(f"{arg_name}: {arg_value}\n")
        f.write("==============================\n")
    
    print(f"参数已记录到 {log_file_path}")

def shuffle_channels_vectorized(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    # 将每个通道展平并拼接
    flat = x.reshape(B * C, -1)
    # 对每个通道独立打乱
    indices = torch.argsort(torch.rand(B * C, H * W, device=x.device), dim=1)
    flat_shuffled = torch.gather(flat, 1, indices)
    # 恢复原始形状
    return flat_shuffled.reshape(B, C, H, W)


def shuffle_all_elements(x: torch.Tensor) -> torch.Tensor:
    """
    随机打乱张量中的所有元素（不考虑维度结构，完全随机化）
    
    参数:
        x: 输入张量，形状为 (B, C, H, W)
    
    返回:
        torch.Tensor: 打乱后的张量，形状不变
    """
    # 展平为一维向量
    flat = x.reshape(-1)
    # 随机打乱所有元素
    perm = torch.randperm(flat.size(0))
    flat_shuffled = flat[perm]
    # 恢复原始形状
    return flat_shuffled.reshape(x.shape)

def embed_secret_in_latent_ldpc(secret_length, total_info_size, paras, target_block_size, backup_r, backup_d, p_error,latent_shape=(4, 64, 64), place_mode=None, margin=1.0):
    """
    将水印信息嵌入到一个 latent 空间中，复制backup_d份数据，复制backup_r份冗余位，其余位置填充随机噪声。

    """ 
    G, H, actual_k, n, d_v, d_c = paras
    block_size =actual_k
    secret = np.random.randint(0,2,size=secret_length)
    secret_backuped = np.tile(secret,backup_d)
    if backup_r>0:
        encoded ,actual_k, n, d_v, d_c, H = encode_with_pyldpc(secret,paras,target_block_size,p_error)
        encoded_backup = np.tile(encoded, backup_r)

        # check
        # print(f'len of secret_backuped = {len(secret_backuped)}')
        # print(f'len of parity_backup = {len(parity_backup)}')
        # sys.exit()

        # 填充剩余的位
        s = total_info_size-len(encoded_backup)
        padding = np.random.randint(0,2,size=s)
        ecc_np = np.concatenate((encoded_backup, padding))
    # else:
    #     parity_len=0
    #     parity_flat_np=None
    #     s = total_info_size-len(secret_backuped)
    #     padding = np.random.randint(0,2,size=s)
    #     ecc_np = np.concatenate((secret_backuped,padding))


    ecc_insert =generate_numbers_from_secret(ecc_np,margin)

    # 计算 latent 总的元素数
    total_positions = np.prod(latent_shape)
    info_length = len(ecc_insert)
    # print(f'>>> info_length ={info_length}')

    if place_mode =='PLACE_LINSPACE':
        positions = np.linspace(0, total_positions - 1, num=info_length, dtype=int)
    elif place_mode == 'PLACE_SEQUENTIAL':
        positions = np.arange(0,info_length)
    elif place_mode == 'PLACE_CHANNEL_SEQ':
        a = latent_shape[1]*latent_shape[2]
        first_channel_positions = np.arange(0,info_length // latent_shape[0])
        positions = np.concatenate((first_channel_positions, first_channel_positions + a), axis=0)
        for i in range(latent_shape[0]-2):
            positions = np.concatenate((positions, first_channel_positions + a*(i+2)), axis=0)
    else:
        raise ValueError("Place mode is unknown.")

    # 生成一个含随机噪声的 latent 张量（这里使用标准正态分布，可以根据需求调整）
    latent = np.random.randn(*latent_shape).astype(np.float32)
    
    # 将 latent 展平成一维数组便于替换secret
    latent_flat = latent.flatten()
    
    # 检查长度是否超出指定位置数
    if info_length > len(positions):
        raise ValueError("secret 的长度超过了所提供的固定位置数量。")
    

    # print(f'positions len = {len(positions)}')
    # print(f'secret len={len(secret)},r len={parity_len},backup_r len = {len(parity_backup)},padding len={len(padding)}, s+br+p ={len(data_padded)+len(parity_backup)+len(padding)}')
    # sys.exit()


    # 在固定位置插入 secret 内容
    latent_flat[positions] = ecc_insert
    
    # 重新 reshape 成指定的 latent 形状
    latent = latent_flat.reshape(latent_shape)
    
    return secret, positions, latent, len(encoded)

