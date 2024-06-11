import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


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
        x1 = self.resnet.layer1(x0) # 256
        x2 = self.resnet.layer2(x1) # 512
        x3 = self.resnet.layer3(x2) # 1024
        x4 = self.resnet.layer4(x3) # 2048
#         x4 = self.conv2(x4)
        
        
        return x4, [x3, x2, x1, x]


class conv_block(nn.Module):
    
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

        
    def forward(self, inputs):
        return self.double_conv(inputs)



class decoder(nn.Module): # upsampling + concat + conv_block
    def __init__(self):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(256+1024, 512) # 1024 from encoder + 1024 from skip
        self.c2 = conv_block(512+512, 256) # 512 from encoder + 512 from skip
        self.c3 = conv_block(256+256, 64) # 256 from encoder + 256 from skip
        self.c4 = conv_block(64+64, 32)    # 64 from encoder + 64 from skip
        
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
    
# class bottle_conv(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.bottle_conv = nn.Sequential(
#             nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#         )
    
#     def forward(self, x):
#         result = self.bottle_conv(x)
#         return result
    

class U_Net_ResNet50(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        
        self.e1 = encoder_ResNet()
        self.d1 = decoder()
        self.b1 = conv_block(2048, 256)
        self.out = nn.Conv2d(32, out_c, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x0 = x
        x, skips = self.e1(x)
        x = self.b1(x)
        x = self.d1(x, skips) 
        x = self.out(x)
        # x = self.sigmoid(x)
        
        return x