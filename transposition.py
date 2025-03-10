from dataclasses import dataclass

from tick_tack_toe import TicTacToeState


@dataclass
class TranspositionTable:
    table = {}  # Dictionary to store nodes indexed by state hash

    def reset(self):
        self.table = {}

    def get_node(self, state: TicTacToeState):
        """Get node from table or return None if not found"""
        # Create a hashable representation of the state
        state_hash = self._state_to_hash(state)
        return self.table.get(state_hash)

    def store_node(self, node):
        """Store node in the table"""
        state_hash = self._state_to_hash(node.state)
        self.table[state_hash] = node

    @staticmethod
    def _state_to_hash(state) -> tuple:
        """Convert state to a hashable representation"""
        # Convert board to a tuple of tuples (which is hashable)
        board_tuple = tuple(tuple(row) for row in state.board)
        return board_tuple, state.current_player


transposition_ctx = TranspositionTable()
