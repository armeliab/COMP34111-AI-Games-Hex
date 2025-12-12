from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired
import sys
import time

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class _MCTSRaveBaseAgent(AgentBase):
    """Base class for RAVE MCTS agents handling process logic."""
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result._colour = self._colour
        result._binary_path = self._binary_path
        result._rave_k = getattr(self, "_rave_k", 300.0)
        result.agent_process = None
        return result

    def __init__(self, colour: Colour):
        super().__init__(colour)
        bin_dir = Path(__file__).with_name("bin")

        if sys.platform == "win32":
            self._binary_path = bin_dir / "mcts_rave_agent.exe"
        else:
            self._binary_path = bin_dir / "mcts_rave_agent"

        self.agent_process = None

    def _launch_process(self, board_size: int):
        if not self._binary_path.exists():
            raise FileNotFoundError(f"Missing binary: {self._binary_path}")

        colour_char = 'R' if self.colour == Colour.RED else 'B'

        self.agent_process = Popen(
    [
        str(self._binary_path),
        colour_char,
        str(board_size),
        str(self._rave_k),
        sys.executable,  # python
        str(Path(__file__).with_name("bin") / "nn_server.py"),
        str(Path(__file__).with_name("bin") / "weights.pt"),
        "60",   # VALUE_SWITCH_EMPTY
        "1.5",  # C_PUCT
    ],
    stdin=PIPE, stdout=PIPE, text=True
)


    def _board_to_string(self, board: Board) -> str:
        return ",".join(
            "".join('R' if t.colour == Colour.RED else
                    'B' if t.colour == Colour.BLUE else '0'
                    for t in row)
            for row in board.tiles
        )

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        if self.agent_process is None:
            self._launch_process(board.size)

        if opp_move is None:
            cmd = f"START;;{self._board_to_string(board)};{turn};"
        elif opp_move.is_swap():
            cmd = f"SWAP;;{self._board_to_string(board)};{turn};"
        else:
            cmd = f"CHANGE;{opp_move.x},{opp_move.y};{self._board_to_string(board)};{turn};"

        self.agent_process.stdin.write(cmd + "\n")
        self.agent_process.stdin.flush()

        resp = self.agent_process.stdout.readline().strip()
        x, y = map(int, resp.split(","))
        return Move(x, y)

    def __del__(self):
        try:
            if self.agent_process:
                self.agent_process.terminate()
        except Exception:
            pass


# ────────────────────────────────────────────────
#     TWO PUBLIC CLASSES: baseline + test agent
# ────────────────────────────────────────────────

class MCTSRaveAgent(_MCTSRaveBaseAgent):
    """Variable-K RAVE agent (uses env var RAVE_K)"""
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._rave_k = float(os.getenv("RAVE_K", "300"))


class MCTSRave300Agent(_MCTSRaveBaseAgent):
    """Fixed-K baseline (K = 300)"""
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._rave_k = 300.0



        # g++ -O3 -std=c++17 agents/Group34/cpp/MCTSRaNNAgent.cpp -o agents/Group34/bin/mcts_rave_nn_agent
