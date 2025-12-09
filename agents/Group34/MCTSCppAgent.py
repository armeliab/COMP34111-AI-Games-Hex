from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired
import sys
import time

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTSCppAgent(AgentBase):
    """
    Thin Python wrapper that forwards board states to the compiled C++ MCTS agent.
    
    The C++ process is launched lazily in make_move to avoid multiprocessing
    serialization issues (pickle/deepcopying a Popen object).
    """

    def __deepcopy__(self, memo):
        """
        Prevent deepcopy from trying to duplicate the subprocess handle.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Copy only the colour, skip the process handle
        result._colour = self._colour
        result._binary_path = self._binary_path
        result.agent_process = None
        return result

    def __init__(self, colour: Colour):
        super().__init__(colour)
        bin_dir = Path(__file__).with_name("bin")
        if sys.platform == "win32":
            self._binary_path = bin_dir / "mcts_cpp_agent.exe"
        else:
            self._binary_path = bin_dir / "mcts_cpp_agent"
        self.agent_process: Popen | None = None
        # NOTE: Do NOT launch the process here! Launch it lazily in make_move.
        # This prevents issues when the game engine deepcopies the agent.

    def _launch_process(self, board_size: int):
        """Initializes the C++ subprocess."""
        if not self._binary_path.exists():
            # Check for the binary file in the expected path
            raise FileNotFoundError(
                f"Compiled agent not found at {self._binary_path}. "
                "Build it with a command similar to: "
                "g++ agents/Group34/cpp/MCTSCppAgent.cpp -O3 -std=c++17 -o agents/Group34/bin/mcts_cpp_agent"
            )

        # Get simple colour character for C++ (R or B)
        colour_char = 'R' if self.colour == Colour.RED else 'B'
        
        # Launch the C++ process, passing colour and board size as arguments
        self.agent_process = Popen(
            [
                str(self._binary_path),
                colour_char,
                str(board_size),
            ],
            stdout=PIPE,
            stdin=PIPE,
            text=True,
        )

    def _board_to_string(self, board: Board) -> str:
        """Converts the Python Board object into the required serialized string format."""
        board_strings = []
        for row in board.tiles:
            row_string = ""
            for tile in row:
                colour = tile.colour
                # Convert to simple characters: R, B, or 0 (no ANSI codes)
                if colour == Colour.RED:
                    row_string += 'R'
                elif colour == Colour.BLUE:
                    row_string += 'B'
                else:
                    row_string += '0'
            board_strings.append(row_string)
        return ",".join(board_strings)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        # **LAZY LAUNCH**: Launch the C++ process only when make_move is called for the first time
        if self.agent_process is None:
            self._launch_process(board.size)

        board_string = self._board_to_string(board)

        # --- Construct the Command String ---
        if opp_move is None:
            # Red's first move
            command_type = "START"
            move_coords = ""
        elif opp_move.is_swap():
            # Player 1 (Red) chose to SWAP on turn 2
            command_type = "SWAP"
            move_coords = ""
        else:
            # Normal move or Player 2 (Blue) deciding to SWAP/CHANGE on turn 2
            command_type = "CHANGE"
            move_coords = f"{opp_move.x},{opp_move.y}"

        # Format: "COMMAND;MOVE;BOARD;TURN;"
        command = f"{command_type};{move_coords};{board_string};{turn};"

        assert self.agent_process is not None  # for type-checkers
        
        # Check if the process has terminated unexpectedly
        if self.agent_process.poll() is not None:
            # Read stdout/stderr for diagnostic info before raising
            stdout_data, stderr_data = self.agent_process.communicate() 
            raise RuntimeError(
                f"C++ agent process terminated unexpectedly. "
                f"STDOUT: {stdout_data.strip()}, STDERR: {stderr_data.strip()}"
            )

        # Send command to C++ agent's stdin
        try:
            self.agent_process.stdin.write(command + "\n")
            self.agent_process.stdin.flush()
        except OSError as e:
            raise RuntimeError(f"Failed to write to C++ agent stdin: {e}") from e

        # Read response from C++ agent's stdout
        response = self.agent_process.stdout.readline().strip()
        
        if not response:
            raise RuntimeError("C++ agent returned an empty response.")

        try:
            x_str, y_str = response.split(",")
            return Move(int(x_str), int(y_str))
        except ValueError as exc:
            # This catches issues with the response format (e.g., non-integers, wrong delimiter)
            raise RuntimeError(f"Malformed response from C++ agent: {response}") from exc

    def __del__(self):
        """Ensure the C++ process is terminated when the Python object is deleted."""
        if hasattr(self, 'agent_process') and self.agent_process is not None:
            try:
                # Close stdin to signal termination if needed
                if self.agent_process.stdin:
                    self.agent_process.stdin.close()
                # Attempt to terminate gracefully
                self.agent_process.terminate()
                # Wait for process to exit
                self.agent_process.wait(timeout=1)
            except TimeoutExpired:
                # Force kill if it doesn't terminate in time
                self.agent_process.kill()
            except Exception:
                # Ignore errors during cleanup
                pass