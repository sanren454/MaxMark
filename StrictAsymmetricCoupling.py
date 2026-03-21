from FrEIA import framework as Fr
from FrEIA import modules as Fm
import torch
import sys
class StrictAsymmetricCoupling(Fm.InvertibleModule):
    def __init__(self, dims_in, dims_c=None, subnet1=None, subnet2=None):
        super().__init__(dims_in,dims_c)
        # 检查子网络是否提供
        if subnet1 is None or subnet2 is None:
            raise ValueError("必须提供subnet1和subnet2参数")
        self.subnet1 = subnet1  # 用于变换x2→y2
        self.subnet2 = subnet2  # 用于变换x1→y1

        self.split_len = dims_in[0][0] // 2  # 输入通道数的一半
        
    def output_dims(self, input_dims):
        # 输出维度与输入维度相同
        return input_dims
    
    def forward(self, x, rev=False, jac=True):
        # 输入处理
        x = x[0]  # 提取主输入张量
        B = x.size(0)   # Batch

        # 确保在通道维度分割（dim=1）
        x1 = x[:, :self.split_len, :, :]  # 形状: (B, split_len, H, W)
        x2 = x[:, self.split_len:, :, :]   # 形状: (B, C-split_len, H, W)

        if not rev:
            # 正向变换
            s1, t1 = self.subnet1(x1).chunk(2, dim=1)
            y2 = x2 * torch.exp(s1) + t1
            s2, t2 = self.subnet2(y2).chunk(2, dim=1)
            y1 = x1 * torch.exp(s2) + t2
            # log_j = s1.sum() + s2.sum() if jac else 0.0
            if jac:
                log_j = s1.flatten(1).sum(1) + s2.flatten(1).sum(1)  # (B,)
            else:
                log_j = x.new_zeros(B)      # or None
            # print(f's1 = {s1}, t1 ={t1}')
        else:
            # 反向变换
            s2, t2 = self.subnet2(x2).chunk(2, dim=1)
            y1 = (x1 - t2) * torch.exp(-s2)
            s1, t1 = self.subnet1(y1).chunk(2, dim=1)
            y2 = (x2 - t1) * torch.exp(-s1)
            # log_j = (-s1).sum() + (-s2).sum() if jac else 0.0
            # print(f's2 = {s2}, t2 ={t2}')
            if jac:
                log_j = (-s1).flatten(1).sum(1) + (-s2).flatten(1).sum(1)
            else:
                log_j = x.new_zeros(B)

        # 合并结果
        out = torch.cat([y1, y2], dim=1)  # 沿通道维度合并
        return (out,), log_j