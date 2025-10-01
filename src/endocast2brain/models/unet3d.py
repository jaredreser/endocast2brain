import torch
import torch.nn as nn

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1),
        nn.InstanceNorm3d(cout),
        nn.ReLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1),
        nn.InstanceNorm3d(cout),
        nn.ReLU(inplace=True),
    )

class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = conv_block(cin, cout)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cin//2, 2, stride=2)
        self.conv = conv_block(cin, cout)
    def forward(self, x, skip):
        x = self.up(x)
        # Pad if needed (when dimensions don’t match perfectly)
        dz = skip.shape[2] - x.shape[2]
        dy = skip.shape[3] - x.shape[3]
        dx = skip.shape[4] - x.shape[4]
        x = nn.functional.pad(x, (0,dx,0,dy,0,dz))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class TinyUNet3D(nn.Module):
    """
    Very small 3D U-Net variant for 64^3 inputs and batch sizes 1–2 on ~8GB VRAM.
    """
    def __init__(self, in_channels=1, out_channels=1, base=8):
        super().__init__()
        self.inc = conv_block(in_channels, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.bot = conv_block(base*4, base*8)
        self.up2 = Up(base*8, base*4)
        self.up1 = Up(base*4, base*2)
        self.outc = nn.Conv3d(base*2, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        xb = self.bot(x3)
        x = self.up2(xb, x3)
        x = self.up1(x, x2)
        x = self.outc(x)
        return x
