import torch
import torch.nn as nn
from layers.unet_blocks import ConvBlock, UpBlock

class DiscriminativeNet(nn.Module):
    def __init__(self, in_channels=6, base_channels=64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels*2)
        self.enc3 = ConvBlock(base_channels*2, base_channels*4)
        
        self.dec3 = UpBlock(base_channels*4, base_channels*2)
        self.dec2 = UpBlock(base_channels*2, base_channels)
        self.dec1 = nn.Conv2d(base_channels,1,kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1,2))
        e3 = self.enc3(nn.functional.max_pool2d(e2,2))
        
        d3 = self.dec3(e3, e2)
        d2 = self.dec2(d3, e1)
        out = torch.sigmoid(self.dec1(d2))  
        return out
