import torch
import os
import torch.nn as nn

# Define SRResNet and ResidualBlock classes
class SRResNet(nn.Module):
    def __init__(self):
        super(SRResNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size= 9, stride= 1, padding= 4), 
            nn.PReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(64)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(64)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(64)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(64)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(64)
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, stride= 1, padding= 1),
            nn.BatchNorm2d(64)
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels = 256, kernel_size= 3, stride= 1, padding= 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.block9 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels = 256, kernel_size= 3, stride= 1, padding= 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.block10 = nn.Conv2d(in_channels = 64, out_channels= 3, kernel_size = 9, stride = 1, padding= 4)
        
    def forward(self, x):
        y = self.block1(x)
        x = torch.add(y, self.block2(y))
        x = torch.add(x, self.block3(x))
        x = torch.add(x, self.block4(x))
        x = torch.add(x, self.block5(x))
        x = torch.add(x, self.block6(x))
        x = torch.add(y, self.block7(x))
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        return x
