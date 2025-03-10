from copy import deepcopy
from typing import Any, List, Tuple

import numpy as np


class TicTacToeState:
    def __init__(self):
        # 3x3 board, 0 = empty, 1 = X, 2 = O
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player 1 (X) starts
        self.winner = None  # None = game in progress, 0 = draw, 1 = X wins, 2 = O wins
        self.moves: list[list[str, tuple]] = []
        self.symbols = {0: " ", 1: "X", 2: "O"}

    def get_legal_moves(self) -> List[Tuple[int, int]] | List[Any]:
        """Return list of empty positions (legal moves)"""
        if self.winner is not None:
            return []  # Game is over, no legal moves

        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    moves.append((i, j))
        return moves

    def make_move(self, move: tuple[int, int]) -> "TicTacToeState":
        """Make a move on the board and return new game state"""
        i, j = move
        if self.board[i, j] != 0:
            raise ValueError("Invalid move, position already taken")

        # Create a new state (don't modify the current one)
        new_state = deepcopy(self)
        new_state.board[i, j] = self.current_player

        # Switch player for next turn
        new_state.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1

        # Check if the game is over
        new_state._check_winner()

        return new_state

    def _check_winner(self) -> None:
        """Check if there's a winner or if it's a draw"""
        # Check rows
        for i in range(3):
            if (
                self.board[i, 0] != 0
                and self.board[i, 0] == self.board[i, 1] == self.board[i, 2]
            ):
                self.winner = self.board[i, 0]
                return

        # Check columns
        for j in range(3):
            if (
                self.board[0, j] != 0
                and self.board[0, j] == self.board[1, j] == self.board[2, j]
            ):
                self.winner = self.board[0, j]
                return

        # Check diagonals
        if (
            self.board[0, 0] != 0
            and self.board[0, 0] == self.board[1, 1] == self.board[2, 2]
        ):
            self.winner = self.board[0, 0]
            return

        if (
            self.board[0, 2] != 0
            and self.board[0, 2] == self.board[1, 1] == self.board[2, 0]
        ):
            self.winner = self.board[0, 2]
            return

        # Check for draw (board is full)
        if len(self.get_legal_moves()) == 0:
            self.winner = 0  # Draw
            return

    def is_terminal(self) -> bool:
        """Check if the game is over"""
        return self.winner is not None

    def get_reward(self, player: int) -> int:
        """Get reward for a player (-1 = loss, 0 = draw, 1 = win)"""
        if self.winner is None:
            return 0  # Game not over yet
        if self.winner == 0:
            return 0  # Draw
        if self.winner == player:
            return 1  # The player won
        return -1  # The player lost

    def __str__(self) -> str:
        """Representation of the board"""
        # symbols = {0: ' ', 1: 'X', 2: 'O'}
        result = "-" * 13 + "\n"
        for i in range(3):
            result += "| "
            for j in range(3):
                result += self.symbols[self.board[i, j]] + " | "
            result += "\n" + "-" * 13 + "\n"
        return result[:-1]  # Getting rid of the last \n
