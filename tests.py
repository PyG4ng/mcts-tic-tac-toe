import unittest

import numpy as np

from mcts import mcts_search
from node import MCTSNode
from tic_tac_toe import TicTacToeState
from transposition import transposition_ctx


class MCTSTestSuite(unittest.TestCase):
    def setUp(self):
        transposition_ctx.reset()
        self.empty_state = TicTacToeState()

    def test_win_in_one(self):
        """Test that MCTS finds a winning move when one exists"""
        # A state where player 1 can win in one move
        state = TicTacToeState()
        # X | X |
        # O |   |
        # O |   |
        state.board = np.array([[1, 1, 0], [2, 0, 0], [2, 0, 0]])

        # Player 1's turn
        state.current_player = 1

        # MCTS should find the winning move (0, 2)
        best_move = mcts_search(state, iterations=1000)
        self.assertEqual(best_move, (0, 2), "MCTS failed to find the winning move")

    def test_block_opponent_win(self):
        """Test that MCTS blocks an opponent's winning move"""
        # A state where player 2 playing with O would win next turn if not blocked
        state = TicTacToeState()
        # X | O | X
        # X | O |
        # O |   |
        state.board = np.array([[1, 2, 1], [1, 2, 0], [2, 0, 0]])

        # Player 1's turn (X should block at 2,1)
        state.current_player = 1

        best_move = mcts_search(state, iterations=1000)
        self.assertEqual(
            best_move, (2, 1), "MCTS failed to block opponent's winning move"
        )

    def test_exploit_forced_win(self):
        """Test that MCTS finds a forced win in multiple moves"""
        state = TicTacToeState()
        # X | O |
        # X | O |
        #   |   |
        state.board = np.array([[1, 2, 0], [1, 2, 0], [0, 0, 0]])

        # Player 1's turn (should choose 2,0 to set up a forced win)
        state.current_player = 1

        best_move = mcts_search(state, iterations=2000)
        self.assertEqual(
            best_move, (2, 0), "MCTS failed to find the move leading to a forced win"
        )

    def test_first_move_preference(self):
        """Test that MCTS prefers strong opening moves"""
        # On an empty board, MCTS should prefer center or corners
        state = self.empty_state
        best_move = mcts_search(state, iterations=3000)

        # Center or corners are theoretically strongest first moves
        strong_first_moves = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
        self.assertIn(
            best_move,
            strong_first_moves,
            f"First move {best_move} is not among the theoretically strongest options",
        )

    def test_node_selection(self):
        """Test that UCB1 selection works correctly"""
        # Create a root node
        root = MCTSNode(self.empty_state)

        # Manually create and add three children with controlled stats
        child1_state = root.state.make_move((0, 0))
        child1 = MCTSNode(child1_state, parent=root, move=(0, 0))
        child1.visits = 10
        child1.results = {1: 8, 2: 1}

        child2_state = root.state.make_move((1, 1))
        child2 = MCTSNode(child2_state, parent=root, move=(1, 1))
        child2.visits = 100
        child2.results = {1: 60, 2: 30}

        child3_state = root.state.make_move((2, 2))
        child3 = MCTSNode(child3_state, parent=root, move=(2, 2))
        child3.visits = 5
        child3.results = {1: 2, 2: 2}

        root.children = [child1, child2, child3]
        root.untried_moves = []  # Mark as fully expanded
        root.visits = child1.visits + child2.visits + child3.visits

        # With low exploration weight, should select child with highest win rate
        selected = root.select_child(exploration_weight=0.1)
        self.assertEqual(
            selected.move, (0, 0), "Failed to select child with highest win rate"
        )

        # With high exploration weight, should select least visited child
        selected = root.select_child(exploration_weight=10)
        self.assertEqual(
            selected.move,
            (2, 2),
            "Failed to select least visited child with high exploration weight",
        )

    def test_backpropagation(self):
        """Test that statistics are properly backpropagated"""
        root = MCTSNode(self.empty_state)
        child_state = root.state.make_move((0, 0))
        child = MCTSNode(child_state, parent=root, move=(0, 0))
        root.children = [child]

        # Backpropagate a result
        result = {1: 1, 2: 0}
        child.update(result)
        child.parent.update(result)

        # Check that visit counts and results were updated
        self.assertEqual(child.visits, 1)
        self.assertEqual(child.results[1], 1)
        self.assertEqual(root.visits, 1)
        self.assertEqual(root.results[1], 1)

    def test_draw_outcome(self):
        """Test that MCTS handles draws correctly"""
        # Create a state where the next move forces a draw
        state = TicTacToeState()
        # X | O | X
        # O | X | X
        # O |   | O
        state.board = np.array([[1, 2, 1], [2, 1, 1], [2, 0, 2]])

        # Player 1's turn
        state.current_player = 1

        # Only legal move is (2, 1) which results in a draw
        best_move = mcts_search(state, iterations=500)
        self.assertEqual(best_move, (2, 1), "The only remaining move is (2, 1)")

        # Make the move and verify it's a draw
        final_state = state.make_move(best_move)
        self.assertEqual(final_state.winner, 0, "Game should end in a draw")

    def test_transposition_table(self):
        """Test that the transposition table works correctly"""
        # Create a state
        state1 = TicTacToeState()
        state1 = state1.make_move((0, 0))  # X in top-left
        state1 = state1.make_move((1, 1))  # O in center
        state1 = state1.make_move((1, 0))  # X in middle-left
        state1 = state1.make_move((2, 0))  # O in bottom-left

        # Create the same state through different moves
        state2 = TicTacToeState()
        state2 = state2.make_move((1, 0))  # X in middle-left
        state2 = state2.make_move((2, 0))  # O in bottom-left
        state2 = state2.make_move((0, 0))  # O in center
        state2 = state2.make_move((1, 1))  # X in top-left

        # Create nodes
        node1 = MCTSNode(state1)
        transposition_ctx.store_node(node1)

        # Retrieve node for state2
        retrieved_node = transposition_ctx.get_node(state2)

        # Should be the same node
        self.assertIsNotNone(
            retrieved_node, "Failed to retrieve node from transposition table"
        )
        self.assertEqual(
            retrieved_node, node1, "Retrieved incorrect node from transposition table"
        )

    def test_visit_count_consistency(self):
        """Test that parent visit count equals sum of children visit counts plus simulations at parent"""
        # Run MCTS for a few iterations
        mcts_search(self.empty_state, iterations=100)

        # Get the root node from the transposition table
        root = transposition_ctx.get_node(self.empty_state)

        # Calculate sum of children's visits
        children_visits_sum = sum(child.visits for child in root.children)

        # Parent's visits should equal sum of children's visits
        self.assertEqual(
            root.visits,
            children_visits_sum,
            "Root visit count should equal sum of children visit counts",
        )

    def test_full_game_vs_random(self):
        """Test a full game of MCTS vs random play"""
        import random

        # Play a full game: MCTS as player 1 vs. random player
        state = self.empty_state

        while not state.is_terminal():
            if state.current_player == 1:
                # AI player
                move = mcts_search(state, iterations=200)
            else:
                # Random player
                legal_moves = state.get_legal_moves()
                move = random.choice(legal_moves)

            state = state.make_move(move)

        # MCTS as player 1 should never lose to random
        self.assertNotEqual(
            state.winner, 2, "MCTS should not lose to random player when going first"
        )


if __name__ == "__main__":
    unittest.main()
