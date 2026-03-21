# standard imports
import torch
import torch.nn as nn

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.modules import GLOWCouplingBlock,AllInOneBlock
import os,sys
# customized coupling block
from StrictAsymmetricCoupling import *


class Subnet1(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 192, 3, padding=1),
            # nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 2*in_channels, 3, padding=1),
            nn.Tanh(),
        )
    def forward(self,x):
        return self.net(x)

class Subnet2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 192, 3, padding=1),
            # nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 2*in_channels, 3, padding=1), 
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)  



class newWatermark:
    def __init__(self, latent_dim=(4,64,64)) -> None:
        super().__init__()
        self.net_deepth = 12  
        self.inn = Ff.SequenceINN(*latent_dim)
        for k in range(self.net_deepth):
            in_channels = latent_dim[0]//2
            subnet1 = Subnet1(in_channels)
            subnet2 = Subnet2(in_channels)
            self.inn.append(StrictAsymmetricCoupling,subnet1=subnet1,subnet2=subnet2)
