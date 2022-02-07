from audioop import bias
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
        super(Discriminator, self).__init__()

        self.global_branch_layer1 = nn.Sequential(nn.conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2, bias= True, dilation = 1),
                                                    nn.BatchNorm2d(64),
                                                    act(negative_slope= 0.2))

        self.global_branch_layer2 = nn.Sequential(nn.conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias= True, dilation = 1),
                                                    nn.BatchNorm2d(64),
                                                    act(negative_slope= 0.2))

        self.global_branch_layer3 = nn.Sequential(nn.conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias= True, dilation = 1),
                                                    nn.BatchNorm2d(64),
                                                    act(negative_slope= 0.2))

        self.global_branch_layer4 = nn.Sequential(nn.conv2d(256, 512, kernel_size=5, stride=1, padding=2, bias= True, dilation = 1),
                                                    nn.BatchNorm2d(64),
                                                    act(negative_slope= 0.2))

        self.global_branch_layer5 = nn.Sequential(nn.conv2d(512, 512, kernel_size=5, stride=1, padding=2, bias= True, dilation = 1),
                                                    nn.BatchNorm2d(64),
                                                    act(negative_slope= 0.2))

        self.global_branch_layer6 = nn.Sequential(nn.conv2d(512, 512, kernel_size=5, stride=1, padding=2, bias= True, dilation = 1),
                                                    nn.BatchNorm2d(64),
                                                    act(negative_slope= 0.2))
        
        self.global_branch_layer_dense = nn.Linear(512*4*4, 512)



        self.local_branch_layer1 = nn.Sequential(nn.conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2, bias= True, dilation = 1),
                                                    nn.BatchNorm2d(64), 
                                                    act(negative_slope= 0.2))
        
        self.local_branch_layer2 = nn.Sequential(nn.conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias= True, dilation = 1),
                                                    nn.BatchNorm2d(64),
                                                    act(negative_slope= 0.2))

        self.local_branch_layer3 = nn.Sequential(nn.conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias= True, dilation = 1),
                                                    nn.BatchNorm2d(64),
                                                    act(negative_slope= 0.2))

        self.local_branch_layer4 = nn.Sequential(nn.conv2d(256, 512, kernel_size=5, stride=1, padding=2, bias= True, dilation = 1),
                                                    nn.BatchNorm2d(64),
                                                    act(negative_slope= 0.2))

        self.local_branch_layer5 = nn.Sequential(nn.conv2d(512, 512, kernel_size=5, stride=1, padding=2, bias= True, dilation = 1),
                                                    nn.BatchNorm2d(64),
                                                    act(negative_slope= 0.2))
        
        self.local_branch_layer_dense = nn.Linear(512*4*4, 512)

        self.classifier = nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(1024, 1))


    def forward(self, x_local , x_global):
        x_local = self.local_branch_layer1(x_local)
        x_local = self.local_branch_layer2(x_local)
        x_local = self.local_branch_layer3(x_local)
        x_local = self.local_branch_layer4(x_local)
        x_local = self.local_branch_layer5(x_local)
        x_local = x_local.view(x_local.size(0), -1)
        x_local = self.local_branch_layer_dense(x_local)


        x_global = self.global_branch_layer1(x_global)
        x_global = self.global_branch_layer2(x_global)
        x_global = self.global_branch_layer3(x_global)
        x_global = self.global_branch_layer4(x_global)
        x_global = self.global_branch_layer5(x_global)
        x_global = self.global_branch_layer6(x_global)
        x_global = x_global.view(x_global.size(0), -1)
        x_global = self.global_branch_layer_dense(x_global)

        output = self.classifier(torch.cat((x_local, x_global), 1))
        return output

                                    