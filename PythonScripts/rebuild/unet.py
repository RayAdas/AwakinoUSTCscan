import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DownXY(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv3D(in_ch, out_ch)
        self.down = nn.Conv3d(
            out_ch, out_ch,
            kernel_size=(3,3,3),
            stride=(2,2,1),
            padding=1
        )

    def forward(self, x):
        x = self.conv(x)
        x_down = self.down(x)
        return x, x_down

class UpXY(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv3D(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False
        )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()

        self.enc1 = DownXY(in_ch, base_ch)
        self.enc2 = DownXY(base_ch, base_ch*2)
        self.enc3 = DownXY(base_ch*2, base_ch*4)

        self.bottleneck = DoubleConv3D(base_ch*4, base_ch*8)

        self.dec3 = UpXY(base_ch*8, base_ch*4, base_ch*4)
        self.dec2 = UpXY(base_ch*4, base_ch*2, base_ch*2)
        self.dec1 = UpXY(base_ch*2, base_ch, base_ch)

        self.out_conv = nn.Conv3d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        # x: (B,1,H,W,T)
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        x = self.bottleneck(x)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        x = self.out_conv(x)
        return x
