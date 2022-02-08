import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.basemodel import BaseModel
from Models.basenet import BaseNet
import numpy as np
from functools import reduce
from torch.optim import lr_scheduler


class DMFB(nn.Module):
    def __init__(self):
        """insert code here"""
        super(DMFB, self).__init__()
        self.conv_3 = nn.Conv2d(256, 64, 3, 1, 1)
        conv_3_sets = []
        for i in range(4):
            conv_3_sets.append(nn.Conv2d(64, 64, 3, padding=1))
        self.conv_3_sets = nn.ModuleList(conv_3_sets)
        self.conv_3_2 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.conv_3_4 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)
        self.conv_3_8 = nn.Conv2d(64, 64, 3, padding=8, dilation=8)
        self.act_fn = nn.Sequential(nn.ReLU(), nn.InstanceNorm2d(256))
        self.conv_1 = nn.Conv2d(256, 256, 1)
        self.norm = nn.InstanceNorm2d(256)
        
        
    def forward(self, inputs):
        """insert code here"""
        src = inputs
        x = self.act_fn(inputs)
        x = self.conv_3(x)
        K = []
        for i in range(4):
            if i != 0:
                ele = eval('self.conv_3_' + str(2 ** i))(x)
                ele = ele + ele[i - 1]
            else:
                ele = x
            K.append(self.conv_3_sets[i](ele))
        cat = torch.cat(K, 1)
        temp_out = self.conv_1(self.norm(cat))
        out = temp_out + src
        return out
        
        
        

class DFBN(BaseNet):
    def __init__(self):
        """insert code here"""
        super(DFBN, self).__init__()
        noOfDMFB = int(input("Enter the number of times you want the DMFB block to be run."))
        self.basemodel = nn.Sequential(
            nn.Conv2d(4, 64, 5, 1, 2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.InstanceNorm2d(256),
            nn.Sequential(*[DMFB() for _ in range(noOfDMFB)]),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
            )
        
    def forward(self, inputs):
        """insert code here"""
        return self.basemodel(inputs)
        
        
        
class Discriminator(BaseNet):
    def __init__(self, in_channels, cnum=64, is_global=True, act=F.leaky_relu):
        """insert code here"""
        
    def forward(self, x, middle_output=False):
        """insert code here"""