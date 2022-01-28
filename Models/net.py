import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from model.basenet import BaseNet
import numpy as np
from functools import reduce
from torch.optim import lr_scheduler


class DMFB(nn.Module):
    def __init__(self):
        """insert code here"""
        
        
    def forward(self, inputs):
        """insert code here"""
        
        
        

class DFBN(BaseNet):
    def __init__(self):
        """insert code here"""
        
    def forward(self, inputs):
        """insert code here"""
        
        
        
class Discriminator(BaseNet):
    def __init__(self, in_channels, cnum=64, is_global=True, act=F.leaky_relu):
        """insert code here"""
        
    def forward(self, x, middle_output=False):
        """insert code here"""