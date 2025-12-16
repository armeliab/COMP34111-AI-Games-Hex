from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from subprocess import PIPE, Popen

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class DKMCTSRaNNAgent(AgentBase):
    """
    DK wrapper that launches the NN-enabled C++ MCTS+RAVE binary with
    dk_nn_server.py and dk_weights.pt by default.
    """

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result._colour = self._colour
        result._binary_path = self._binary_path
        result._rave_k = self._rave_k
        result._value_switch = self._value_switch
        result._c_puct = self._c_puct
        result._server_path = self._server_path
        result._weights_path = self._weights_path
        result.agent_process = None
        return result

    def __init__(self, colour: Colour):
        super().__init__(colour)
        bin_dir = Path(__file__).with_name("bin")
        if sys.platform == "win32":
            self._binary_path = bin_dir / "dk_mcts_rave_nn_agent.exe"
        else:
            self._binary_path = bin_dir / "dk_mcts_rave_nn_agent"

        self._server_path = bin_dir / "dk_nn_server.py"
        self._weights_path = bin_dir / "dk_weights.pt"
        self._rave_k = 200.0
        self._value_switch = 60
        self._c_puct = 1.2
        self.agent_process: Popen | None = None

    def _launch_process(self, board_size: int):
        if not self._binary_path.exists():
            raise FileNotFoundError(f"Missing binary: {self._binary_path}")
        if not self._server_path.exists():
            raise FileNotFoundError(f"Missing NN server: {self._server_path}")
        if not self._weights_path.exists():
            raise FileNotFoundError(f"Missing weights: {self._weights_path}")

        colour_char = 'R' if self.colour == Colour.RED else 'B'
        # Args: colour, board_size, RAVE_K, python, server, weights, VALUE_SWITCH_EMPTY, C_PUCT
        self.agent_process = Popen(
            [
                str(self._binary_path),
                colour_char,
                str(board_size),
                str(self._rave_k),
                sys.executable,
                str(self._server_path),
                str(self._weights_path),
                str(self._value_switch),
                str(self._c_puct),
            ],
            stdin=PIPE,
            stdout=PIPE,
            text=True,
        )

    def _board_to_string(self, board: Board) -> str:
        return ",".join(
            "".join(
                'R' if t.colour == Colour.RED else
                'B' if t.colour == Colour.BLUE else '0'
                for t in row
            )
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

        assert self.agent_process is not None
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
