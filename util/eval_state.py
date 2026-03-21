import os
import torch
import torch.nn.functional as F
from .utils import embed_secret_in_latent,visualize_latent


def eval_inn_state(inn,discriminator,args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inn.eval()

    with torch.no_grad():
        # 生成一些测试数据进行重构损失计算
        batch_secret = []
        batch_latent = []
        batch_decoded = []
        
        for t in range(args.inn_batch_size):
            secret, positions, latent = embed_secret_in_latent(args.secret_length, args.latent_shape, args.place_mode) 
            secret = torch.tensor(secret).float().to(device)
            latent = torch.tensor(latent).to(device)
            batch_secret.append(secret.unsqueeze(0))
            batch_latent.append(latent.unsqueeze(0))
        
        latent = torch.cat(batch_latent, dim=0).to(device)
        secret = torch.cat(batch_secret, dim=0).to(device)

        z, _ = inn(latent)
        z_noisy = z + torch.randn_like(z) * args.noise_scale
        decoded = inn(z_noisy, rev=True)[0]

        # 只关注水印信息
        for t in range(args.inn_batch_size):
            batch_decoded.append(decoded[t].flatten()[positions].unsqueeze(0))
        decoded = torch.cat(batch_decoded, dim=0) 

        # 二值化
        decoded = (decoded>0).float()

        # 计算mse损失
        loss_mse = F.mse_loss(decoded,secret)
        # 对抗损失
        d_z = discriminator(z)
        loss_adv = -d_z.mean()

        # 计算acc
        is_similar = torch.abs(decoded - secret) < args.threshold
        correct_bits = torch.sum(is_similar).item()/args.inn_batch_size
        acc=correct_bits/args.secret_length

        # 可视化
        visualize_latent(latent=z[0],bin_wid=0.1)

        print(f'>>> evaluating INN, loss_mse = {loss_mse.item()}, loss_adv = {loss_adv.item()}, bit acc = {acc:.4f}')

        return loss_mse.item()+loss_adv.item()
    
def eval_disc_state(inn,disc,args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    disc.eval()
    inn.eval()
    with torch.no_grad():
        # 生成真实样本和生成样本
        z_target = torch.randn(args.disc_batch_size, *args.latent_shape).to(device)
        batch_latent=[]
        for t in range(args.disc_batch_size):
            secret, positions, latent = embed_secret_in_latent(args.secret_length, args.latent_shape, args.place_mode) 
            latent = torch.tensor(latent).to(device)
            batch_latent.append(latent.unsqueeze(0))
        z_fake = torch.cat(batch_latent, dim=0).to(device)
        
        # 计算对抗损失
        # loss_d_real = F.binary_cross_entropy(disc(z_target), torch.ones_like(disc(z_target)))
        # loss_d_fake = F.binary_cross_entropy(disc(z_fake), torch.zeros_like(disc(z_fake)))
        # loss_d = (loss_d_real + loss_d_fake) / 2
        d_real = disc(z_target)
        d_fake = disc(z_fake)
        loss_d = d_fake.mean() - d_real.mean()

        print(f'>>> evaluating discriminator, loss_mse = {loss_d.item()}')

        return loss_d.item()
    
def check_inn_stop(inn, discriminator, best_loss, patience, args):
    eval_loss=eval_inn_state(inn, discriminator, args)
    print(f'INN evaluation loss : {eval_loss:.4f}')
    if not os.path.exists('./models/best_models'):
        os.path.mkdir('./models/best_models')

    if eval_loss < best_loss:
        best_loss = eval_loss
        patience=0
        # 保存当前最佳INN模型
        torch.save(inn.state_dict(),'./models/best_models/best_inn.pth')
    else:
        patience+=1
        return False    # 不停止，继续训练INN
    if patience >= args.patience_threshold:
        print("Early stopping triggered.")
        return True     # 停止训练INN
    
def check_disc_stop(inn, disc, best_loss, patience, args):
    eval_loss=eval_disc_state(inn, disc, args)
    print(f'Discriminator evaluation loss : {eval_loss:.4f}')
    if not os.path.exists('./models/best_models'):
        os.mkdir('./models/best_models')

    if eval_loss < best_loss:
        best_loss = eval_loss
        patience=0
        # 保存当前最佳INN模型
        torch.save(disc.state_dict(),'./models/best_models/best_disc.pth')
    else:
        patience+=1
        return False    # 不停止，继续训练判别器
    if patience >= args.patience_threshold:
        print("Early stopping triggered.")
        return True     # 停止训练判别器