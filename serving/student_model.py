"""Distilled student backbone architecture — lives in serving/ because the
production API depends on it directly. Training code (optimization/distillation.py)
imports the class from here rather than the other way around, so serving never
depends on the research/experiments folder.
"""
import os

import torch.nn as nn
import torch.nn.functional as F

from config import CHECKPOINT_DIR

STUDENT_CHECKPOINT_PATH = CHECKPOINT_DIR / "student_backbone.pt"
OUT_CHANNELS = 384
TARGET_GRID = 74


class StudentBackbone(nn.Module):
    """Small CNN: raw image -> (B, 384, 74, 74), same interface as the DINOv2 backbone."""

    def __init__(self, out_channels: int = OUT_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        feats = self.net(x)
        return F.interpolate(feats, size=(TARGET_GRID, TARGET_GRID), mode="bilinear", align_corners=False)
