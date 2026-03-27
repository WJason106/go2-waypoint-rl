# -*- coding: utf-8 -*-
import torch.nn as nn

class AdaptationModule(nn.Module):
    """estimator / adaptation / ROA 占位模块"""
    def __init__(self, in_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ELU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)