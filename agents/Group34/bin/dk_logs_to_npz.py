#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

RED = 1
BLUE = 2


def parse_move_field(field: str):
    """Parse a move field like 'RED(x=7, y=4)' or 'BLUESWAP()'."""
    field = field.strip()
    colour = RED if field.startswith("RED") else BLUE
    if "SWAP" in field:
        return colour, True, (-1, -1)
    xs = field.split("x=")[1]
    x_str, rest = xs.split(",", 1)
    y_str = rest.split("y=")[1].rstrip(")")
    return colour, False, (int(x_str), int(y_str))


def swap_board(board: np.ndarray):
    swapped = board.copy()
    swapped[board == RED] = BLUE
    swapped[board == BLUE] = RED
    return swapped


def process_log(log_path: Path, board_size: int):
    boards, policies, values, to_moves = [], [], [], []
    colour_by_player = {}
    board = np.zeros((board_size, board_size), dtype=np.int8)
    samples_to_fill = []  # indices whose value is pending final winner

    winner_name = None
    with log_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == "winner":
                winner_name = row[1]
                break
            if not row[0].isdigit():
                continue  # skip summary lines

            if len(row) < 3:
                continue
            player_name = row[1]
            # Move field may be split by the comma between x and y; rejoin columns 2..-2
            if len(row) == 3:
                move_field = row[2]
            else:
                move_field = ",".join(row[2:-1]).strip()
            colour, is_swap, (x, y) = parse_move_field(move_field)

            if player_name not in colour_by_player:
                colour_by_player[player_name] = colour

            if is_swap:
                board = swap_board(board)
                for k, v in list(colour_by_player.items()):
                    colour_by_player[k] = BLUE if v == RED else RED
                continue

            to_move_colour = colour
            plane_me = (board == to_move_colour).astype(np.float32)
            plane_opp = (board == (BLUE if to_move_colour == RED else RED)).astype(np.float32)
            boards.append(np.stack([plane_me, plane_opp], axis=0))

            one_hot = np.zeros((board_size * board_size,), dtype=np.float32)
            idx = x * board_size + y
            one_hot[idx] = 1.0
            policies.append(one_hot)
            to_moves.append(0 if to_move_colour == RED else 1)
            samples_to_fill.append(len(boards) - 1)

            board[x, y] = colour

    if winner_name is None:
        raise RuntimeError(f"No winner line found in {log_path}")
    if winner_name not in colour_by_player:
        raise RuntimeError(f"Winner {winner_name} not seen in moves in {log_path}")
    winner_colour = colour_by_player[winner_name]

    for i in samples_to_fill:
        tm = RED if to_moves[i] == 0 else BLUE
        values.append(1.0 if tm == winner_colour else -1.0)

    return boards, policies, values, to_moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=str, nargs="+", required=True, help="Paths to Hex.py -l CSV logs")
    ap.add_argument("--board", type=int, default=11)
    ap.add_argument("--out", type=str, default="agents/Group34/bin/dk_selfplay.npz")
    args = ap.parse_args()

    all_boards, all_pols, all_vals, all_to = [], [], [], []
    for lp in args.logs:
        b, p, v, t = process_log(Path(lp), args.board)
        all_boards.extend(b)
        all_pols.extend(p)
        all_vals.extend(v)
        all_to.extend(t)

    if not all_boards:
        raise RuntimeError("No samples produced; check your logs.")

    boards_arr = np.stack(all_boards, axis=0).astype(np.float32)   # [B,2,N,N]
    pols_arr = np.stack(all_pols, axis=0).astype(np.float32)       # [B,N*N]
    vals_arr = np.array(all_vals, dtype=np.float32)                # [B]
    to_arr = np.array(all_to, dtype=np.int8)                       # [B]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, boards=boards_arr, policies=pols_arr, values=vals_arr, to_move=to_arr)
    print(f"[dk_logs_to_npz] wrote {boards_arr.shape[0]} samples to {args.out}")


if __name__ == "__main__":
    main()
