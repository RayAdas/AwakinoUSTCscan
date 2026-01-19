"""U-Net architecture for depth map reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, dropout: float = 0.1):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle size mismatch if input dimensions are not multiples of 2
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for depth reconstruction.
    
    Input: (B, wave_len, H, W) - wave signals at each spatial position
    Output: (B, 1, H, W) - reconstructed depth map
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        base_channels: int = 64,
        bilinear: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels, dropout=dropout)
        self.down1 = Down(base_channels, base_channels * 2, dropout=dropout)
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout=dropout)
        
        factor = 2 if bilinear else 1
        self.down3 = Down(base_channels * 4, base_channels * 8 // factor, dropout=dropout)
        
        # Decoder
        self.up1 = Up(base_channels * 8, base_channels * 4 // factor, bilinear, dropout=dropout)
        self.up2 = Up(base_channels * 4, base_channels * 2 // factor, bilinear, dropout=dropout)
        self.up3 = Up(base_channels * 2, base_channels, bilinear, dropout=dropout)
        
        # Output layer
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, wave_len, H, W)
        
        Returns:
            Depth map of shape (B, 1, H, W)
        """
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder path with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
