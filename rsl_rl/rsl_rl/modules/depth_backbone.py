# -*- coding: utf-8 -*-
import torch.nn as nn

class DepthBackbone(nn.Module):
    def __init__(self, in_dim=128, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim),
            nn.ELU(),
        )

    def forward(self, x):
        return self.net(x)