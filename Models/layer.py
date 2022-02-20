import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ### VGG19 Architecture
# 1. Block 1
#     * Conv3x3 (64) #relu1_1
#     * Conv3x3 (64) #relu1_2
#     * MaxPool
#     
# 2. Block 2
#     * Conv3x3 (128) #relu2_1
#     * Conv3x3 (128) #relu2_2
#     * MaxPool
# 3. Block 3
#     
#     * Conv3x3 (256) #relu3_1
#      * Conv3x3 (256) #relu3_2
#     * Conv3x3 (256) #relu3_3
#     * Conv3x3 (256) #relu3_4
#     * MaxPool
#     
# 4. Block 4
#     * Conv3x3 (512) #relu4_1
#     * Conv3x3 (512) #relu4_2
#     * Conv3x3 (512) #relu4_3
#     * Conv3x3 (512) #relu4_4
#     * MaxPool
# 
# 5. Block 5
#     * Conv3x3 (512) #relu5_1
#     * Conv3x3 (512) #relu5_2
#     * Conv3x3 (512) #relu5_3
#     * Conv3x3 (512) #relu5_4
#     * MaxPool
#     
# 6. Block 6
#     * Fully Connected (4096)
#     * Dropout with 50% drop-rate <br>
#     <br>
#     * Fully Connected (4096)
#     * Dropout with 50% drop-rate <br>  
#     <br>
#     * Fully Connected (1000)
#     * SoftMax  
# 
# #### Additional Points:
# - ReLU is used all over
# - All convolutional layers have zero padding with stride=1
# - MaxPool layers have kernel size=2x2 and stride=2

# In[2]:


class VGG19(nn.Module):
    def __init__(self,pool='max'):
        super(VGG19, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding='same')
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding='same')
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding='same')
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding='same')
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding='same')
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding='same')
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding='same')
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding='same')
        
        if pool=='max':
                self.pool = nn.MaxPool2d(2, stride=2)
        else: 
            self.pool = nn.AvgPool2d(2, stride=2)
        self.relu_maps = {}
        
    
    def forward(self,x):
        self.relu_maps['r11'] = F.relu(self.conv1_1(x))
        self.relu_maps['r12'] = F.relu(self.conv1_2(self.relu_maps['r11']))
        pooled_output = self.pool(self.relu_maps['r12'])
        
        self.relu_maps['r21'] = F.relu(self.conv2_1(pooled_output))
        self.relu_maps['r22'] = F.relu(self.conv2_2(self.relu_maps['r21']))
        pooled_output = self.pool(self.relu_maps['r22'])
        
        self.relu_maps['r31'] = F.relu(self.conv3_1(pooled_output))
        self.relu_maps['r32'] = F.relu(self.conv3_2(self.relu_maps['r31']))
        self.relu_maps['r33'] = F.relu(self.conv3_3(self.relu_maps['r32']))
        self.relu_maps['r34'] = F.relu(self.conv3_4(self.relu_maps['r33']))
        pooled_output = self.pool(self.relu_maps['r34'])
        
        self.relu_maps['r41'] = F.relu(self.conv4_1(pooled_output))
        self.relu_maps['r42'] = F.relu(self.conv4_2(self.relu_maps['r41']))
        self.relu_maps['r43'] = F.relu(self.conv4_3(self.relu_maps['r42']))
        self.relu_maps['r44'] = F.relu(self.conv4_4(self.relu_maps['r43']))
        pooled_output = self.pool(self.relu_maps['r44'])
        
        self.relu_maps['r51'] = F.relu(self.conv5_1(pooled_output))
        self.relu_maps['r52'] = F.relu(self.conv5_2(self.relu_maps['r51']))
        self.relu_maps['r53'] = F.relu(self.conv5_3(self.relu_maps['r52']))
        self.relu_maps['r54'] = F.relu(self.conv5_4(self.relu_maps['r53']))
        return self.relu_maps   

class VGG19Features(nn.Module):
    def __init__(self):
        super(VGG19Features, self).__init__()
        vgg = models.vgg19(pretrained=False)
        state_dict = torch.load(r'F:\python\DeepLearning\hhub\DFBM\inpainting_gmcnn\pytorch\checkpoints\vgg19-dcbb9e9d.pth')
        vgg.load_state_dict(state_dict)
        self.vgg19 = vgg.features.eval().cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        out = {}
        x = x - self.mean
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            x = layer(x)
            if isinstance(layer, nn.ReLU) and ri == 1:
                out[name] = x
                if ci == 5:
                    break
        # print([x for x in out])
        return out