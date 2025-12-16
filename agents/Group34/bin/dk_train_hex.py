from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dk_network import DKHexNet, make_coord_planes  # noqa: E402


class HexDataset(Dataset):
    """
    Expects an .npz with:
      boards: float32 [B, 2, N, N] (me, opp)
      policies: float32 [B, N*N]   (probabilities or visit counts)
      values: float32 [B]          (in [-1,1], from to-move perspective)
      to_move: int8 [B]            (0 for Red, 1 for Blue)  # optional; ignored if already aligned
    """
    def __init__(self, npz_path: Path):
        z = np.load(npz_path)
        self.boards = z["boards"]
        self.policies = z["policies"]
        self.values = z["values"]
        self.to_move = z["to_move"] if "to_move" in z else None
        self.n = int(math.sqrt(self.policies.shape[1]))

    def __len__(self):
        return self.boards.shape[0]

    def __getitem__(self, idx):
        b = self.boards[idx]  # [2,N,N]
        p = self.policies[idx]
        v = self.values[idx]
        if self.to_move is not None:
            # ensure plane0 is to-move, plane1 is opp
            if self.to_move[idx] == 1:  # Blue to move vs Red
                b = b[[1, 0]]  # swap planes
                v = -v
        return b, p, v


def add_coords(batch_boards: torch.Tensor):
    # batch_boards: [B,2,N,N] -> [B,4,N,N]
    bsz, _, n, _ = batch_boards.shape
    device = batch_boards.device
    rc, cc = make_coord_planes(n, device)
    coords = torch.stack([rc, cc], dim=0).unsqueeze(0).expand(bsz, -1, -1, -1)
    return torch.cat([batch_boards, coords], dim=1)


def train(args):
    device = (
        torch.device("mps")
        if (args.device is None and torch.backends.mps.is_available())
        else torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[dk_train] device: {device}")

    ds = HexDataset(Path(args.data))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    model = DKHexNet(board_size=ds.n, channels=args.channels, blocks=args.blocks, dropout=0.05).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_loss = 1e9
    out_ckpt = Path(args.out)
    out_ts = Path(args.out_ts)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_pol = 0.0
        total_val = 0.0
        steps = 0

        for boards, tgt_p, tgt_v in dl:
            boards = boards.to(device)
            tgt_p = tgt_p.to(device)
            tgt_v = tgt_v.to(device).float()

            # normalize policy targets (if they are counts)
            tgt_p = tgt_p / torch.clamp(tgt_p.sum(dim=1, keepdim=True), min=1e-6)

            x = add_coords(boards)  # [B,4,N,N]
            logits, v = model(x)
            v = v.flatten()

            pol_loss = F.kl_div(
                F.log_softmax(logits, dim=1),
                tgt_p,
                reduction="batchmean",
            )
            val_loss = F.mse_loss(v, tgt_v)
            loss = pol_loss + args.val_weight * val_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_pol += pol_loss.item()
            total_val += val_loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        print(
            f"[epoch {epoch:03d}] loss={avg_loss:.4f} pol={total_pol/steps:.4f} "
            f"val={total_val/steps:.4f} time={time.time()-t0:.1f}s"
        )

        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), out_ckpt)
            # also export TorchScript for C++ if needed
            example = torch.zeros((1, 4, ds.n, ds.n), device=device)
            ts = torch.jit.trace(model, example)
            ts.save(str(out_ts))
            print(f"  saved best -> {out_ckpt}, {out_ts}")

    print("done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="npz with boards, policies, values")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--val-weight", type=float, default=1.0)
    ap.add_argument("--device", type=str, default=None, help="mps|cuda|cpu (default: auto)")
    ap.add_argument("--out", type=str, default="agents/Group34/bin/dk_weights.pt")
    ap.add_argument("--out-ts", type=str, default="agents/Group34/bin/dk_hex_ts.pt")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
