from __future__ import annotations
import torch
import torch.nn as nn


class HexNet(nn.Module):
    def __init__(self, board_size: int, channels: int = 64, layers: int = 4):
        super().__init__()
        self.n = board_size
        self.head_dim = board_size * board_size

        feats = []
        in_ch = 2
        for i in range(layers):
            feats.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=False))
            feats.append(nn.BatchNorm2d(channels))
            feats.append(nn.ReLU(inplace=True))
            in_ch = channels
        self.feats = nn.Sequential(*feats)

        # Policy head
        self.pol = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.pol_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value head
        self.val = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.val_fc1 = nn.Linear(1 * board_size * board_size, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        # x: [B,2,N,N]
        f = self.feats(x)

        p = self.pol(f).flatten(1)
        logits = self.pol_fc(p)  # [B, N*N]

        v = self.val(f).flatten(1)
        v = torch.relu(self.val_fc1(v))
        v = torch.tanh(self.val_fc2(v))  # [-1,1]
        return logits, v
