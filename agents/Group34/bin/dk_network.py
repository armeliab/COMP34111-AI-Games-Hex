from __future__ import annotations
import torch
import torch.nn as nn


def make_coord_planes(n: int, device: torch.device):
    # 2 coord planes normalized to [0,1]
    r = torch.linspace(0, 1, steps=n, device=device).view(n, 1).expand(n, n)
    c = torch.linspace(0, 1, steps=n, device=device).view(1, n).expand(n, n)
    return r, c


class DKResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.f(x))


class DKHexNet(nn.Module):
    """
    Stronger Hex net: coord channels + residual tower.
    Default: 10 blocks, 96 channels — trainable on M4 Pro (MPS).
    """
    def __init__(self, board_size: int, channels: int = 96, blocks: int = 10, dropout: float = 0.05):
        super().__init__()
        self.n = board_size
        in_ch = 4  # 2 stone planes + 2 coord planes

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[DKResidualBlock(channels) for _ in range(blocks)])

        # Policy head
        self.pol_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.pol_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Value head
        self.val_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.val_fc1 = nn.Linear(board_size * board_size, 256)
        self.val_drop = nn.Dropout(dropout)
        self.val_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: [B,4,N,N] (stones + coords already concatenated)
        f = self.tower(self.stem(x))

        p = self.pol_head(f).flatten(1)
        logits = self.pol_fc(p)

        v = self.val_head(f).flatten(1)
        v = torch.relu(self.val_fc1(v))
        v = self.val_drop(v)
        v = torch.tanh(self.val_fc2(v))
        return logits, v
