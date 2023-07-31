import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, 2 * out_channels, kernel_size=kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(2 * out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x += residual

        return x