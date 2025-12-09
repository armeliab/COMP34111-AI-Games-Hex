import math
from random import choice
from time import time

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.AgentBase import AgentBase


class MCTSAgentMuhammad(AgentBase):
    
    def __init__(self, colour):
        super().__init__(colour)
    
    def make_move(self, turn: int, board, opp_move) -> Move:
        moves = self.get_all_valid_moves(board, turn)
        root = self.MCTSNode(board, self.colour, moves)
        
        return root.get_best_move(2)

    @staticmethod
    def get_all_valid_moves(board, turn = -1):
        BOARD_SIZE = 11
        moves = [
            Move(i, j)
            for i in range(BOARD_SIZE)
            for j in range(BOARD_SIZE)
            if board.tiles[i][j].colour is None
        ]
        if turn == 2:
            moves.append(Move(-1, -1))
        return moves
    
    @staticmethod
    def clone_board(board):
        new_board = Board(11)
        for i in range(board.size):
            for j in range(board.size):
                new_board.set_tile_colour(i, j, board.tiles[i][j].colour)
        return new_board
    
    class MCTSNode:
        def __init__(self, board, colour, moves, associated_move=None, parent=None):
            self.board: Board = board
            self.parent: MCTSAgentMuhammad.MCTSNode | None = parent
            self.colour: Colour = colour
            self.moves: list[Move] = moves
            self.associated_move: Move = associated_move
            self.children: list[MCTSAgentMuhammad.MCTSNode] = []
            self.wins = 0
            self.visits = 0
            
        def select_child(self):
            
            if not self.is_fully_expanded():
                return self
            
            def ucb1(child):
                if child.visits == 0:
                    return float('inf')
                exploitation = child.wins / child.visits
                exploration = math.sqrt(math.log(self.visits) / child.visits)
                return exploitation + 1.4 * exploration
            
            return max(self.children, key=ucb1).select_child()
        
        def best_move(self):
            best_child = max(self.children, key=lambda c: c.visits)
            return best_child

        def get_best_move(self, seconds: float = 1.0) -> Move:
            
            current_time = time()
            
            while not self.is_fully_expanded():
                self.expand().simulate_random_playout()
                
            while time() - current_time < seconds:
                node = self.select_child()
                node.expand().run_simulations()
            
            return self.best_move().associated_move

        def is_fully_expanded(self):
            return self.moves == []
        
        def expand(self):
            move = self.moves.pop()
            new_board = self.simulate_move(self.board, move)
            child_node = MCTSAgentMuhammad.MCTSNode(new_board, Colour.opposite(self.colour), MCTSAgentMuhammad.get_all_valid_moves(new_board), move, parent=self)
            self.children.append(child_node)
            return child_node
        
        def simulate_move(self, board, move):
            new_board = MCTSAgentMuhammad.clone_board(board)
            new_board.set_tile_colour(move.x, move.y, self.colour)
            return new_board
        
        def simulate_random_playout(self):
            current_board = MCTSAgentMuhammad.clone_board(self.board)
            current_colour = self.colour

            while not self._is_game_ended(current_board):
                possible_moves = MCTSAgentMuhammad.get_all_valid_moves(current_board)
                move = choice(possible_moves)
                current_board.set_tile_colour(move.x, move.y, current_colour)
                current_colour = Colour.opposite(current_colour)

            winner = current_board.get_winner()
            self.backpropagate(winner)
            
        
        def _is_game_ended(self, board: Board) -> bool:
            """Safely check if the game has ended for either colour.

            Board.has_ended expects a Colour; calling it without a colour
            raises ValueError. Treat any ValueError as "not ended" so the
            playout continues.
            """
            try:
                return board.has_ended(Colour.RED) or board.has_ended(Colour.BLUE)
            except ValueError:
                return False

        def backpropagate(self, winner):
            root = self
            while root.parent is not None:
                root = root.parent
            
            node = self
            while node is not None:
                node.visits += 1
                if root.colour == winner:
                    node.wins += 1
                node = node.parent
                
        def run_simulations(self):
            for _ in range(10):
                self.simulate_random_playout()
            
    
