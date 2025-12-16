from __future__ import annotations
import argparse
import sys
import torch
import torch.nn.functional as F

from dk_network import DKHexNet, make_coord_planes  # noqa: E402


def parse_board(board_str: str, board_size: int, to_move: str, device: torch.device) -> torch.Tensor:
    """
    Returns tensor [1, 4, N, N]:
    plane0: stones of to-move
    plane1: stones of opponent
    plane2: row coords in [0,1]
    plane3: col coords in [0,1]
    """
    rows = board_str.split(",")
    n = board_size
    assert len(rows) == n, f"expected {n} rows, got {len(rows)}"

    me = to_move
    opp = "B" if me == "R" else "R"

    x = torch.zeros((1, 4, n, n), dtype=torch.float32, device=device)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == me:
                x[0, 0, r, c] = 1.0
            elif ch == opp:
                x[0, 1, r, c] = 1.0
    rc, cc = make_coord_planes(n, device)
    x[0, 2] = rc
    x[0, 3] = cc
    return x


@torch.inference_mode()
def eval_pos(model, board_str: str, board_size: int, to_move: str, device: torch.device):
    x = parse_board(board_str, board_size, to_move, device)
    logits, value = model(x)
    logits = logits[0]
    value = value.reshape(-1)[0].clamp(-1, 1)
    policy = F.softmax(logits, dim=0)
    return float(value.item()), policy.detach().cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=int, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default=None, help="mps|cuda|cpu (default: auto)")
    # Defaults match the trained DK model (128 channels, 12 blocks).
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=12)
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    n = args.board
    model = DKHexNet(board_size=n, channels=args.channels, blocks=args.blocks)
    sd = torch.load(args.model, map_location="cpu")
    model.load_state_dict(sd)
    model.eval().to(device)

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
            p_csv = ",".join(f"{x:.8g}" for x in p)
            print(f"OK;{v:.6f};{p_csv}", flush=True)
        except Exception as e:
            print(f"ERR;{type(e).__name__}", flush=True)


if __name__ == "__main__":
    main()
