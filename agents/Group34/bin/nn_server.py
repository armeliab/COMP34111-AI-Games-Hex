from __future__ import annotations

import argparse
import sys
import torch
import torch.nn.functional as F

from network import HexNet  # you provide this


def parse_board(board_str: str, board_size: int, to_move: str) -> torch.Tensor:
    """
    Returns tensor shape [1, 2, N, N]
    Plane 0: stones of player-to-move
    Plane 1: stones of opponent
    """
    rows = board_str.split(",")
    n = board_size
    assert len(rows) == n, f"Expected {n} rows, got {len(rows)}"

    # Map chars to 2 planes
    # to_move in {"R","B"}
    me = to_move
    opp = "B" if me == "R" else "R"

    x = torch.zeros((1, 2, n, n), dtype=torch.float32)
    for r in range(n):
        row = rows[r]
        for c in range(n):
            ch = row[c]
            if ch == me:
                x[0, 0, r, c] = 1.0
            elif ch == opp:
                x[0, 1, r, c] = 1.0
    return x


@torch.inference_mode()
def eval_pos(model: torch.nn.Module, board_str: str, board_size: int, to_move: str, device: torch.device):
    x = parse_board(board_str, board_size, to_move).to(device)
    logits, value = model(x)  # logits: [1, N*N], value: [1, 1] or [1]
    logits = logits[0]
    value = value.reshape(-1)[0].clamp(-1, 1)

    policy = F.softmax(logits, dim=0)
    return float(value.item()), policy.detach().cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=int, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    n = args.board

    model = HexNet(board_size=n)
    sd = torch.load(args.model, map_location=device)
    model.load_state_dict(sd)
    model.eval().to(device)

    # Line protocol
    # PING -> PONG
    # EVAL;R;<board> -> OK;<value>;<p0>,...,<pN-1>
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "PING":
            print("PONG", flush=True)
            continue

        if not line.startswith("EVAL;"):
            print("ERR;bad_command", flush=True)
            continue

        try:
            _, to_move, board_str = line.split(";", 2)
            v, p = eval_pos(model, board_str, n, to_move, device)
            # CSV policy (fast for C++ parser)
            p_csv = ",".join(f"{x:.8g}" for x in p)
            print(f"OK;{v:.6f};{p_csv}", flush=True)
        except Exception as e:
            print(f"ERR;{type(e).__name__}", flush=True)


if __name__ == "__main__":
    main()
