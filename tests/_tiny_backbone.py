"""A tiny image backbone for in-process training tests (backbone_type='custom')."""
import torch
import torch.nn as nn


class TinyBackbone(nn.Module):
    def __init__(self, feat_dim: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(3, feat_dim, kernel_size=4, stride=4)
        self.feat_dim = feat_dim

    def forward(self, x):
        h = torch.relu(self.conv(x))                 # (B, D, h, w)
        patches = h.flatten(2).transpose(1, 2)       # (B, N, D)
        return patches.mean(1), patches              # (cls, patch tokens)
