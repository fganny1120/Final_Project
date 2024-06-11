#!/usr/bin/env python
# coding: utf-8

# In[17]:


'''model'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.models import resnet50, ResNet50_Weights


class conv_block(nn.Module): # for decoder
    
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class Conv2D(nn.Module): # for ASPP
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class ASPP(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        
        self.averagepool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(inputs, outputs, kernel_size=1, padding=0)
        )
        self.c1 = Conv2D(inputs, outputs, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(inputs, outputs, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(inputs, outputs, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(inputs, outputs, kernel_size=3, padding=18, dilation=18)
        
        self.c5 = Conv2D(outputs*5, outputs, kernel_size=1, padding=0, dilation=1)
        
    def forward(self, x):
        x0 = self.averagepool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear", align_corners=True)
        
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        
        xc = torch.cat([x0, x1, x2, x3, x4], axis=1)
        y = self.c5(xc)
        
        return y      
    
class encoder_ResNet(nn.Module):
    def __init__(self):
        super(encoder_ResNet, self).__init__()
        
        self.resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(2048, 2, bias=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.resnet.maxpool = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        self.conv2 = nn.Conv2d(2048, 512, 3, 1, 1)
        
    
    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        
        x0 = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x0)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
#         x4 = self.conv2(x4)
        
        
        return x4, [x3, x2, x1, x]

class decoder1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(512+1024, 512)
        self.c2 = conv_block(512+512, 256)
        self.c3 = conv_block(256+256, 128)
        self.c4 = conv_block(128+64, 64)
        
    def forward(self, x, skip):
        s1, s2, s3, s4 = skip
        
        x = self.up(x)
        x = torch.cat([x, s1], axis=1)
        x = self.c1(x)
        
        x = self.up(x)
        x = torch.cat([x, s2],axis=1)
        x = self.c2(x)
        
        x = self.up(x)
        x = torch.cat([x, s3],axis=1)
        x = self.c3(x)
        
        x = self.up(x)
        x = torch.cat([x, s4],axis=1)
        x = self.c4(x)
        
        return x
    
    
class U_Net_ResNet50_ASPP(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        
        self.e1 = encoder_ResNet()
        self.a1 = ASPP(2048, 512)
        self.d1 = decoder1()
        self.y1 = nn.Conv2d(64, out_c, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x0 = x
        x, skip1 =self.e1(x)
        x = self.a1(x)
        x = self.d1(x, skip1)
        y1 = self.y1(x)
        return y1
    
if __name__ == "__main__":
    x = torch.randn((1, 3, 512, 512))
    f = U_Net_ResNet50_ASPP()
    y = f(x)
    
    print(y.shape)




