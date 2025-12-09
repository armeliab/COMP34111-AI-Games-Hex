from random import choice

from src.Move import Move
from src.AgentBase import AgentBase


class MCTSAgentMuhammad(AgentBase):
    
    _board_size: int = 11

    def __init__(self, colour):
        super().__init__(colour)
    
    def make_move(self, turn: int, board, opp_move) -> Move:
        return choice(self.get_all_valid_moves(board, turn))

    def get_all_valid_moves(self, board, turn):
        moves = [
            Move(i, j)
            for i in range(self._board_size)
            for j in range(self._board_size)
            if board.tiles[i][j].colour is None
        ]
        if turn == 2:
            moves.append(Move(-1, -1))
        return moves

    
