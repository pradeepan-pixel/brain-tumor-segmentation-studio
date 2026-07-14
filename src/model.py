"""
model.py — 3D U-Net sized for consumer GPUs (RTX 3050, 4-6GB VRAM).

Key changes vs a "textbook" 3D UNet:
  - InstanceNorm3d instead of BatchNorm3d: with patch-based training your
    effective batch size is 1-2, and BatchNorm statistics are unstable/broken
    at that batch size. InstanceNorm (normalize per-sample) is the nnU-Net
    standard for exactly this reason.
  - LeakyReLU instead of ReLU: standard nnU-Net choice, avoids dead neurons.
  - Configurable base_channels (default 16, not 32) to keep activation memory
    manageable on small VRAM budgets. Bump to 32 if you have more VRAM headroom.
  - Optional gradient checkpointing: trades compute for memory by not storing
    intermediate activations, recomputing them during backward instead. Turn
    this on if you still OOM at your target patch size.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):

    def __init__(self, in_channels=4, out_channels=1, base_channels=16,
                 use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # Encoder
        self.enc1 = DoubleConv(in_channels, c1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(c3, c4)

        # Decoder
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c4, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c3, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c2, c1)

        # Output (1 channel = binary whole-tumor mask; use raw logits, apply
        # sigmoid in the loss / at inference, not here)
        self.final = nn.Conv3d(c1, out_channels, kernel_size=1)

    def _run_block(self, block, x):
        if self.use_checkpointing and self.training:
            return cp.checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x):
        # Encoder
        e1 = self._run_block(self.enc1, x)
        p1 = self.pool1(e1)

        e2 = self._run_block(self.enc2, p1)
        p2 = self.pool2(e2)

        e3 = self._run_block(self.enc3, p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self._run_block(self.bottleneck, p3)

        # Decoder
        u3 = self.up3(b)
        u3 = torch.cat((u3, e3), dim=1)
        d3 = self._run_block(self.dec3, u3)

        u2 = self.up2(d3)
        u2 = torch.cat((u2, e2), dim=1)
        d2 = self._run_block(self.dec2, u2)

        u1 = self.up1(d2)
        u1 = torch.cat((u1, e1), dim=1)
        d1 = self._run_block(self.dec1, u1)

        out = self.final(d1)
        return out


if __name__ == "__main__":
    # Quick sanity check — run this file directly to confirm shapes work
    # at your intended patch size before launching real training.
    model = UNet3D(in_channels=4, out_channels=1, base_channels=16)
    x = torch.randn(1, 4, 96, 96, 96)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
