import sys
from tqdm import tqdm
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from util.utils import *
from newWm_v4 import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import numpy as np
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
import PIL
from PIL import Image, ImageFilter,ImageEnhance
import cv2
import matplotlib.pyplot as plt


@torch.no_grad()
def eval_inn_disc(inn, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    secret, positions, latent = embed_secret_in_latent(secret_length=args.secret_length, place_mode=args.place_mode)
    secret = torch.tensor(secret).float().to(device)
    latent = torch.tensor(latent).unsqueeze(0).to(device)
    inn.eval()
    # discriminator.eval()

    z, log_jac_det = inn(latent) 
    # 可视化
    visualize_latent(z,bin_wid=0.1)

    decoded, _ = inn(z, rev=True)

    # d_z = discriminator(z)
    # d_target = discriminator(z_target)

    # 检查是否是恒等变换
    # is_similar = torch.abs(decoded[0] - latent[0]) < args.threshold
    # correct_bits = torch.sum(is_similar).item()  
    # print(f'total latent similar bits = {correct_bits}')
    # sys.exit()

    decoded = decoded.flatten()[positions]
    # print(f'secret shape={secret.shape}\ndecoded shape = {decoded.shape}')
    # sys.exit()
    # 二值化
    decoded = (decoded >0).float()

    is_similar = torch.abs(decoded - secret) < args.threshold
    correct_bits = torch.sum(is_similar).item()    

    acc = correct_bits/args.secret_length
    print(f'>>> correct_bits = {correct_bits}')


def train_inn_with_SD(inn,pipe,dataset,args,epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer_inn = torch.optim.Adam(inn.parameters(), lr=1e-5, betas=(0.5, 0.999))
    
    inn.train()
    # 保存路径
    save_path =f'{args.save_path}/{args.secret_length}_{args.margin}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 记录损失
    reconstruct_loss, mle_dis_loss, moment_dis_loss = [],[],[]

    for i in tqdm(range(epochs)):
        secret, positions, latent = embed_secret_in_latent(args.secret_length,args.latent_shape,args.place_mode,args.margin) 
        secret = torch.tensor(secret).float().unsqueeze(0).to(device)   # np array -> tensor (float32) 
        
        latent = torch.tensor(latent).unsqueeze(0).to(device)

        # INN forward
        z, log_jac_det = inn(latent)
        z_target = torch.randn_like(z)
        loss_mmd = channel_wise_mmd_loss(z, z_target)
        loss_mle = mle_loss(z,log_jac_det)
        loss_moment = moment_loss(z)
        loss_channel_mom = channel_moment_loss(z)
        
        print(f'loss state:\n分布损失：\nloss_mle = {loss_mle}\nloss_moment = {loss_channel_mom}')


        loss_total = loss_mle+ 0.1*loss_channel_mom
        
        # 反向传播
        optimizer_inn.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(inn.parameters(), 1.0)
        optimizer_inn.step()
    


        if (i+1) % 10 == 0:
            if args.place_mode is not None:
                torch.save(inn.state_dict(),f'{save_path}/pretrained_inn_{args.place_mode}_epoch{i}.pth')
            else:
                torch.save(inn.state_dict(),f'{save_path}/pretrained_inn_epoch{i}.pth')
            # print(f"Epoch {epoch}: Loss Total={loss_total.item():.3f}, MSE={loss_mse.item():.3f}, Adv={loss_adv.item():.3f}")
    if args.place_mode is not None:
        torch.save(inn.state_dict(),f'{save_path}/pretrained_inn_{args.place_mode}_final.pth')
    else:
        torch.save(inn.state_dict(),f'{save_path}/pretrained_inn_final.pth')


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_printoptions(sci_mode=False,profile='full')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 引入INN
    inn =newWatermark(args.latent_shape).inn.to(device)
    if not os.path.exists(f'{args.save_path}/{args.secret_length}_{args.margin}'):
        os.mkdir(f'{args.save_path}/{args.secret_length}_{args.margin}')
                          
    torch.save(inn.state_dict(),f'{args.save_path}/{args.secret_length}_{args.margin}/pretrain_0.pth')
    if args.model !=None:
        inn.load_state_dict(torch.load(args.model))
    train_inn_with_SD(inn,None,None,args,args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test if INN can be robust to SD model')
    parser.add_argument('--latent_shape',default=(4,64,64))
    # parser.add_argument('--latent_shape',default=(8,250,16))
    # parser.add_argument('--latent_shape',default=(8,750,32))
    # parser.add_argument('--latent_shape',default=(4,16,64,64))
    parser.add_argument('--secret_length', default=16384, type=int)
    parser.add_argument('--place_mode',default='PLACE_SEQUENTIAL')
    parser.add_argument('--threshold',default=0.1,type=float)
    parser.add_argument('--model', default=None, type=str,help='INN model')
    parser.add_argument('--save_path',default='train_outputs')
    parser.add_argument('--epochs',default=400,type=int)
    parser.add_argument('--margin',default=1.0,type=float)
    args =parser.parse_known_args()[0]

    main(args)