from copy import deepcopy

from node import MCTSNode
from tic_tac_toe import TicTacToeState
from transposition import transposition_ctx


def mcts_search(state, iterations=1000, exploration_weight=1.41) -> tuple[int, int]:
    """Run MCTS algorithm and return the best move"""
    root = MCTSNode(state)
    transposition_ctx.reset()
    transposition_ctx.store_node(root)

    for _ in range(iterations):
        # Phase 1: Selection - Select a promising node
        node = root
        while not node.is_terminal_node() and node.is_fully_expanded():
            node = node.select_child(exploration_weight)

        # Phase 2: Expansion - Expand the tree with a new node
        if not node.is_terminal_node() and not node.is_fully_expanded():
            node = node.expand()

        # Phase 3: Simulation - Run a simulation from the new node
        simulation_state = smart_rollout(node.state)

        # # Random playout until terminal state
        # simulation_state = deepcopy(node.state)
        # while not simulation_state.is_terminal():
        #     # Choose a random move
        #     random_move = random.choice(simulation_state.get_legal_moves())
        #     simulation_state = simulation_state.make_move(random_move)

        # Calculate result
        result = {1: simulation_state.get_reward(1), 2: simulation_state.get_reward(2)}

        # Phase 4: Backpropagation - Update statistics up the tree
        while node is not None:
            node.update(result)
            node = node.parent

    # Select the move with the highest win rate
    current_player = state.current_player
    best_child = max(
        root.children, key=lambda child: child.results[current_player] / child.visits
    )

    return best_child.move


def smart_rollout(state: TicTacToeState) -> TicTacToeState:
    """Running a simulation with a smarter policy instead of random moves"""
    simulation_state = deepcopy(state)

    while not simulation_state.is_terminal():
        current_player = simulation_state.current_player
        opponent = 3 - current_player
        # First check if current player can win immediately
        for move in simulation_state.get_legal_moves():
            temp_state = deepcopy(simulation_state)
            test_state = temp_state.make_move(move)
            if test_state.winner == current_player:
                return test_state  # Found a winning move

        # Next check if opponent has a winning move that needs to be blocked
        for move in simulation_state.get_legal_moves():
            temp_state = deepcopy(simulation_state)
            # switch the current player to the opponent to simulate his move
            temp_state.current_player = opponent
            test_state = temp_state.make_move(move)
            if test_state.winner == opponent:
                # Opponent could win with this move, we should block it
                return test_state
        else:  # No blocking needed
            # Choose moves based on a fixed hierarchy
            # Try to take the center, then the corners in a fixed order otherwise take any remaining edge
            center = (1, 1)
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
            for move in [center] + corners + edges:
                if move in simulation_state.get_legal_moves():
                    return simulation_state.make_move(move)

    return simulation_state
