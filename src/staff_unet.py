"""
staff_unet.py — Small U-Net for staff line segmentation.

3-channel BGR input → single-channel staff line mask output.

Architecture: standard U-Net, 64→128→256→512→1024 channels, 4 down-up
pairs.  Roughly 31M parameters — fits comfortably on a 3050 with batch
size 2 at 512×512, batch size 8 at 256×256.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    """Two 3×3 convs with BN+ReLU after each."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Standard U-Net for binary segmentation.

    Output is raw logits — wrap in sigmoid for probabilities.  Pair with
    BCEWithLogitsLoss (or DiceBCELoss) during training.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()

        self.down1 = _DoubleConv(in_channels, 64)
        self.down2 = _DoubleConv(64, 128)
        self.down3 = _DoubleConv(128, 256)
        self.down4 = _DoubleConv(256, 512)
        self.middle = _DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(2)

        self.up4   = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv4 = _DoubleConv(1024, 512)
        self.up3   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = _DoubleConv(512, 256)
        self.up2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = _DoubleConv(256, 128)
        self.up1   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = _DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        m  = self.middle(self.pool(d4))

        u4 = self.up4(m)
        u4 = torch.cat([u4, d4], dim=1)
        u4 = self.conv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.out(u1)


class DiceBCELoss(nn.Module):
    """
    Combined BCE + Dice loss.

    Pure BCE underweights the rare positive class (staff line pixels are
    typically <1% of the image).  Dice fixes that by working on overlap
    fraction.  The combination tends to converge faster and more stably
    than either alone.
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return bce_loss + (1.0 - dice)


if __name__ == '__main__':
    # Sanity check
    model = UNet()
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'U-Net  input {tuple(x.shape)}  ->  output {tuple(y.shape)}')
    print(f'Parameters: {n_params:,}  ({n_params * 4 / 1e6:.1f} MB float32)')
