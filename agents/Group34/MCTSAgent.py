from random import choice
import math
from random import random

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTSAgent(AgentBase):
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.

    The class inherits from AgentBase, which is an abstract class.
    The AgentBase contains the colour property which you can use to get the agent's colour.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    _choices: list[Move]
    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]
        self.tree = None

    @staticmethod
    def clone_board(board: Board) -> Board:
        new = Board(board.size)
        for i in range(board.size):
            for j in range(board.size):
                new_tile = new.tiles[i][j]
                old_tile = board.tiles[i][j]
                new_tile.colour = old_tile.colour
        return new
    
    class MCTSNode:
        def __init__(self, state: Board, player: Colour, root_colour, parent=None, action=None):
            self.state = state
            self.player = player
            self.root_colour = root_colour
            self.parent = parent
            self.action = action
            self.children = []
            self.visits = 0
            self.wins = 0
            self.untried_actions = self.get_actions()

        def get_actions(self):
  
            board = self.state
            return [
                (i, j)
                for i in range(board.size)
                for j in range(board.size)
                if (board.tiles[i][j]).colour is None
            ]

        def is_terminal(self):
            """Check if the game has ended"""
            return self.check_winner() is not None or not self.get_actions()
        
        def is_fully_expanded(self):
            """Check is there are any more actions left"""
            return len(self.untried_actions) == 0
        
        def check_winner(self):
            board = self.state
            if board.has_ended(Colour.RED):
                return Colour.RED
            if board.has_ended(Colour.BLUE):
                return Colour.BLUE
            return None
        

        
        
        def expand(self):
            action = self.untried_actions.pop()

            new_state = MCTSAgent.clone_board(self.state)

            x, y = action
            new_state.set_tile_colour(x, y, self.player)

            next_player = Colour.opposite(self.player)
            child = MCTSAgent.MCTSNode(new_state, next_player, self.root_colour, parent=self, action=action)
            self.children.append(child)
            return child
        
        
        def best_child(self, c=1.4):
            """Select child with best UCB1 score."""
            def ucb(child):

                if child.visits == 0:
                    return float("inf")
                
                return ((child.wins / child.visits) +
                         c * math.sqrt(math.log(self.visits + 1) / child.visits))

            return max(self.children, key=ucb)
        
        def rollout(self):
            state = MCTSAgent.clone_board(self.state)
            player = self.player

            def neighbours(x, y, size):
                pos = [
                        (-1, 0), (1, 0),
                        (0, -1), (0, 1),
                        (-1, 1), (1, -1)
                    ]
                for dx, dy in pos:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        yield nx, ny

            def biased_random_action(state, player, actions):
                epsilon = 0.1
                if random() < epsilon:
                    return choice(actions)
                a = 3.0  # bias for own neighbours
                b = 1.0  # bias for opponent neighbours

                scores = []
                opp = Colour.opposite(player)

                for (x, y) in actions:
                    n_self = 0
                    n_opp = 0

                    for nx, ny in neighbours(x, y, state.size):
                        c = state.tiles[nx][ny].colour
                        if c is None:
                            continue
                        if c == player:
                            n_self += 1
                        elif c == opp:
                            n_opp += 1

                    score = 1.0 + a * n_self + b * n_opp
                    scores.append(score)
            
                # If all scores are 0 (shouldn't happen with the +1, but just in case)
                total = sum(scores)
                if total <= 0:
                    return choice(actions)
                
                # Sample proportional to score (simple roulette-wheel selection)
                r = random() * total
                acc = 0.0
                for (x, y), w in zip(actions, scores):
                    acc += w
                    if r <= acc:
                        return x, y
                
                # Fallback (numerical edge case)
                return actions[-1]

    
            while True:
                if state.has_ended(Colour.RED):
                    winner = Colour.RED
                elif state.has_ended(Colour.BLUE):
                    winner = Colour.BLUE
                else:
                    winner = None

                if winner is not None:
                    return 1.0 if winner == self.root_colour else 0.0
                

                actions = [
                            (i, j)
                            for i in range(state.size)
                            for j in range(state.size)
                            if (state.tiles[i][j]).colour is None
                        ]
                
                if not actions: return 0.5

                x, y = biased_random_action(state, player, actions)
                state.set_tile_colour(x, y, player)

                player = Colour.opposite(player)

        def backpropagate(self, result):
            self.visits += 1
            self.wins += result
            if self.parent:
                self.parent.backpropagate(result)
                
        
    def mcts_search(self, root: "MCTSAgent.MCTSNode", iterations=100) -> Move:

        for _ in range(iterations):
            # print("Iteration ->",_)
            node = root

            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()
            
            # Expansion
            if not node.is_terminal():
                node = node.expand()
            
            # TODO:
            # - add rollout (simulation)
            # - add backpropagation of result

            # Simulation
            result = node.rollout()

            # Backpropagation
            node.backpropagate(result)
        
        # For now, just pick a legal move from the root as a fallback
        # actions = root.get_actions()
        # if not actions:
        #     # should never happen in a non-terminal game, but just in case
        #     return Move(0, 0)

        # x, y = choice(actions)
        best_child = root.best_child(c=0)
        x, y = best_child.action
        return Move(x, y)


    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """
        
        # if turn == 2 and choice([0, 1]) == 1:
        if turn == 2:
            return Move(-1, -1)
        else:
            player = self.colour
            
            if self.tree is None:
                self.tree = MCTSAgent.MCTSNode(board, player, self.colour)

            elif opp_move is not None and (opp_move.x, opp_move.y) != (-1, -1):
                for child in self.tree.children:
                    if child.action == (opp_move.x, opp_move.y):
                        self.tree = child
                        self.tree.parent = None
                        break
            
            best_move = self.mcts_search(self.tree, iterations=100)

            for child in self.tree.children:
                if child.action == (best_move.x, best_move.y):
                    self.tree = child
                    self.tree.parent = None
                    break

            return best_move

    #  python3 Hex.py -p1 "agents.Group34.MCTSAgent MCTSAgent" -p2 "agents.MCTSAgent.MCTSAgent MCTSAgent"