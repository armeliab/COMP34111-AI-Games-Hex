import math
from random import choice
from time import time

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.AgentBase import AgentBase


class MCTSAgent(AgentBase):
    
    _root: 'MCTSAgent.MCTSNode | None' = None
    
    def __init__(self, colour):
        super().__init__(colour)
    
    def make_move(self, turn: int, board, opp_move) -> Move:
        moves = self.get_pruned_moves(board, turn)
        root = self.MCTSNode(board, self.colour, moves)

        
        return root.get_best_move(1)

    def update_root(self, turn: int, board, opp_move):
        if self._root == None:
            return MCTSAgent.MCTSNode(board, self.colour, self.get_all_valid_moves(board, turn))
        else:
            for child in self._root.children:
                if child.associated_move == opp_move:
                    child.parent = None
                    return child

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
    
    @staticmethod
    def has_nearby_stone(board: Board, move: Move) -> bool:
        x, y = move.x, move.y
        size = board.size
        # Hex 6-neighbour offsets
        offsets = [(-1, 0), (1, 0),
                   (0, -1), (0, 1),
                   (-1, 1), (1, -1)]
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if board.tiles[nx][ny].colour is not None:
                    return True
        return False

    @staticmethod
    def get_pruned_moves(board: Board, turn=-1):
        # If early game, don’t prune too aggressively
        empty_count = sum(
            1 for i in range(board.size)
              for j in range(board.size)
              if board.tiles[i][j].colour is None
        )

        BASE_MOVES = [
            Move(i, j)
            for i in range(board.size)
            for j in range(board.size)
            if board.tiles[i][j].colour is None
        ]

        if turn == 2:
            BASE_MOVES.append(Move(-1, -1))

        # If very early, just return all
        if empty_count > board.size * board.size - 4:
            return BASE_MOVES

        pruned = [
            m for m in BASE_MOVES
            if MCTSAgent.has_nearby_stone(board, m)
        ]

        # Fallback: if everything pruned (weird positions), use all
        return pruned if pruned else BASE_MOVES

    class MCTSNode:
        def __init__(self, board, colour, moves, associated_move=None, parent=None):
            self.board: Board = board
            self.parent: MCTSAgent.MCTSNode | None = parent
            self.colour: Colour = colour
            self.moves: list[Move] = moves
            self.associated_move: Move = associated_move # Move that led to this node
            self.children: list[MCTSAgent.MCTSNode] = []
            self.wins = 0
            self.visits = 0

            # AMAF / RAVE stats: (x,y) -> [wins, visits]
            self.amaf_stats: dict[tuple[int, int], list[int]] = {}
            
        def select_child(self):
            if not self.is_fully_expanded():
                return self

            def score(child: 'MCTSAgent.MCTSNode'):
                if child.visits == 0:
                    return float('inf')

                Q = child.wins / child.visits

                # AMAF value for this move from this node’s perspective
                mv = (child.associated_move.x, child.associated_move.y) if child.associated_move is not None else None
                if mv is not None and mv in self.amaf_stats and self.amaf_stats[mv][1] > 0:
                    amaf_wins, amaf_visits = self.amaf_stats[mv]
                    Q_amaf = amaf_wins / amaf_visits
                else:
                    Q_amaf = 0.5  # neutral

                # Mix factor beta: high when visits small, shrinking as visits grow
                k = 300.0
                beta = k / (self.visits + k)

                mixed_Q = (1 - beta) * Q + beta * Q_amaf

                
                exploration = math.sqrt(math.log(self.visits) / child.visits)
                C = 0.5  # smaller than 1.4 

                return mixed_Q + C * exploration

            return max(self.children, key=score).select_child()

        
        def best_move(self):
            best_child = max(self.children, key=lambda c: c.visits)
            return best_child

        def get_best_move(self, seconds: float = 1.0) -> Move:
            
            current_time = time()
            simulation_count = 0
            while time() - current_time < seconds:
                node = self.select_child()                
                node.expand().simulate_random_playout()
                simulation_count += 1

            print(f"Simulations: {simulation_count}")
            
            return self.best_move().associated_move

        def is_fully_expanded(self):
            return self.moves == []
        
        def expand(self):
            move = self.moves.pop()
            new_board = self.simulate_move(self.board, move)
            child_node = MCTSAgent.MCTSNode(
                        new_board,
                        Colour.opposite(self.colour),
                        MCTSAgent.get_pruned_moves(new_board),
                        move,
                        parent=self
                    )

            self.children.append(child_node)
            return child_node
        
        def simulate_move(self, board, move):
            new_board = MCTSAgent.clone_board(board)
            new_board.set_tile_colour(move.x, move.y, self.colour)
            return new_board
        
        def simulate_random_playout(self):
            current_board = MCTSAgent.clone_board(self.board)
            current_colour = self.colour

            last_move: Move | None = None
            playout_moves: list[tuple[int, int]] = []  # record all moves

            while not self._is_game_ended(current_board):
                possible_moves = MCTSAgent.get_all_valid_moves(current_board)
                if not possible_moves:
                    break

                forced_move = None
                if last_move is not None:
                    forced_move = self._find_bridge_response(current_board, last_move, current_colour)

                if forced_move is not None:
                    move = forced_move
                else:
                    move = choice(possible_moves)

                current_board.set_tile_colour(move.x, move.y, current_colour)
                playout_moves.append((move.x, move.y))

                last_move = move
                current_colour = Colour.opposite(current_colour)

            winner = current_board.get_winner()
            self.backpropagate(winner, playout_moves)


            
        def _find_bridge_response(self, board: Board, last_move: Move, current_colour: Colour) -> Move | None:
            """
            If last_move just probed a bridge belonging to current_colour,
            return the forced reply (the other eye of the bridge), else None.

            We approximate the classic Hex bridge with a diamond in (x,y) coords:
              anchors: (x, y) and (x+1, y+1)
              eyes:    (x, y+1) and (x+1, y)
            and the symmetric one:
              anchors: (x, y) and (x+1, y-1)
              eyes:    (x, y-1) and (x+1, y)
            """
            x, y = last_move.x, last_move.y
            size = board.size

            def in_bounds(i, j):
                return 0 <= i < size and 0 <= j < size

            # List of candidate patterns: each is (eye1, eye2, anchor1, anchor2)
            patterns = []

            # Diagonal / shape up-right
            # eyes: (x, y) and (x+1, y) or (x, y+1)
            # We'll consider last_move as one eye, other_eye computed accordingly
            # Pattern 1: last_move is upper eye of diag up-right: (x, y) = (x, y+1)
            patterns.append((
                (x, y),           # eye1 (where last_move might be)
                (x + 1, y - 1),   # eye2
                (x, y - 1),       # anchor1
                (x + 1, y)        # anchor2
            ))

            # Pattern 2: last_move is lower eye of diag up-right: (x, y) = (x+1, y)
            patterns.append((
                (x, y),           # eye1
                (x - 1, y + 1),   # eye2
                (x - 1, y),       # anchor1
                (x, y + 1)        # anchor2
            ))

            # Diagonal / shape up-left
            # Pattern 3: last_move is upper eye of diag up-left: (x, y) = (x, y-1)
            patterns.append((
                (x, y),           # eye1
                (x + 1, y + 1),   # eye2
                (x, y + 1),       # anchor1
                (x + 1, y)        # anchor2
            ))

            # Pattern 4: last_move is lower eye of diag up-left: (x, y) = (x+1, y)
            patterns.append((
                (x, y),           # eye1
                (x - 1, y - 1),   # eye2
                (x - 1, y),       # anchor1
                (x, y - 1)        # anchor2
            ))

            for eye1, eye2, a1, a2 in patterns:
                ex1, ey1 = eye1
                ex2, ey2 = eye2
                ax1, ay1 = a1
                ax2, ay2 = a2

                if not (in_bounds(ex2, ey2) and in_bounds(ax1, ay1) and in_bounds(ax2, ay2)):
                    continue

                # last_move must be at eye1
                if (x, y) != (ex1, ey1):
                    continue

                # anchors must exist and belong to current_colour
                if board.tiles[ax1][ay1].colour != current_colour:
                    continue
                if board.tiles[ax2][ay2].colour != current_colour:
                    continue

                # other eye must be empty
                if board.tiles[ex2][ey2].colour is None:
                    return Move(ex2, ey2)

            return None

        def _is_game_ended(self, board: Board) -> bool:
            try:
                return board.has_ended(Colour.RED) or board.has_ended(Colour.BLUE)
            except ValueError:
                return False

        def backpropagate(self, winner, playout_moves=None):
            if playout_moves is None:
                playout_moves = []
            root = self
            while root.parent is not None:
                root = root.parent

            node = self
            while node is not None:
                node.visits += 1
                if root.colour == winner:
                    node.wins += 1

                    # AMAF: give this playout’s moves credit as well
                    for (mx, my) in playout_moves:
                        key = (mx, my)
                        if key not in node.amaf_stats:
                            node.amaf_stats[key] = [0, 0]
                        node.amaf_stats[key][0] += 1  # amaf wins
                        node.amaf_stats[key][1] += 1  # amaf visits
                else:
                    # Even if lost, still increment AMAF visits
                    for (mx, my) in playout_moves:
                        key = (mx, my)
                        if key not in node.amaf_stats:
                            node.amaf_stats[key] = [0, 0]
                        node.amaf_stats[key][1] += 1

                node = node.parent
