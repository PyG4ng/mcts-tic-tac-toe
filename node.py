import math
import random

from tic_tac_toe import TicTacToeState
from transposition import transposition_ctx


class MCTSNode:
    def __init__(
        self,
        state: TicTacToeState,
        parent: "MCTSNode" = None,
        move: tuple[int, int] = None,
    ):
        self.state: TicTacToeState = state  # Game state at this node
        self.parent: "MCTSNode" = parent  # Parent node
        self.move: tuple[int, int] = move  # Move that led to this state

        self.children: list["MCTSNode"] = []  # Child nodes
        self.untried_moves: list = state.get_legal_moves()  # Moves not yet explored

        self.visits: int = 0  # Number of times this node has been visited
        self.results: dict = {1: 0, 2: 0}  # Wins for each player

    def __str__(self):
        return f"Children: {self.children} - Move: {self.move}"

    def select_child(self, exploration_weight=1.41) -> "MCTSNode":
        """
        Select a child node using the UCB1 formula:
        UCB1 = (wins / visits) + exploration_weight * sqrt(ln(parent_visits) / visits)
        """
        # Convert current player to the one who will make a move at the child node
        player = self.state.current_player  # The player who just made a move

        # Select child with highest UCB1 value
        # for child in self.children:
        # print(f"{child.move = }", "UCB1" ,(child.results[player] / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits))
        return max(
            self.children,
            key=lambda child: (child.results[player] / child.visits)
            + exploration_weight * math.sqrt(math.log(self.visits) / child.visits),
        )

    def expand(self) -> "MCTSNode":
        """Expand the tree by adding a new child node"""
        # Choose a random untried move
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)

        # Create new child state by applying the move
        child_state = self.state.make_move(move)

        existing_node = transposition_ctx.get_node(child_state)

        if existing_node:
            # Link the existing node as a child of the current node
            self.children.append(existing_node)
            # Update the parent reference
            existing_node.parent = self
            child = existing_node
        else:
            # Create and add child node
            child = MCTSNode(child_state, parent=self, move=move)
            self.children.append(child)
            transposition_ctx.store_node(child)

        return child

    def update(self, result: dict[int, int]) -> None:
        """Update node statistics"""
        self.visits += 1

        # Update win counts for each player
        self.results[1] += result[1]
        self.results[2] += result[2]

    def is_fully_expanded(self) -> bool:
        """Check if all possible moves from this state have been explored"""
        return len(self.untried_moves) == 0

    def is_terminal_node(self) -> bool:
        """Check if this node represents a terminal game state"""
        return self.state.is_terminal()
