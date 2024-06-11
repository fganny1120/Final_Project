import torch
import torch.nn as nn
import torch.nn.functional as F


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

    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        
        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0) #size doubled
        self.conv = conv_block(out_c+out_c, out_c)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x,skip], axis=1)
        x = self.conv(x)
        return x
    
class U_Net(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        
        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64,128)
        self.e3 = encoder_block(128,256)
        self.e4 = encoder_block(256,512)
        
        """ Bottleneck """
        self.b = conv_block(512, 1024)
        
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        
        """ Classifier (FCN) """
        self.outputs = nn.Conv2d(64, out_c ,kernel_size=1, padding=0)
        
    def forward(self,inputs):
        
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        """ Bottleneck """
        b = self.b(p4)
        
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        
        outputs = self.outputs(d4)
        
        return outputs
        
if __name__ == "__main__":
    x = torch.randn((1, 3, 128, 128))
    f = U_Net()
    y = f(x)
    
    print(y.shape)

    x = torch.randn(1, 3, 512, 512)
    f = U_Net(1)
    y = f(x)
    print(y.shape)