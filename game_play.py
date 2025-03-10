from mcts import mcts_search
from tick_tack_toe import TicTacToeState


def play_game():
    """A game of Tic-Tac-Toe: AI vs You"""

    for _ in range(10):
        state = TicTacToeState()
        print("You are playing as O, AI is X")

        while not state.is_terminal():
            if state.current_player == 1:
                print("\nAI is thinking...")
                ai_move = mcts_search(state, iterations=10_000)
                print(ai_move)
                state.moves.append([state.symbols.get(state.current_player), ai_move])
                state = state.make_move(ai_move)
                print(f"AI plays at position {ai_move}")
                print(state)

            else:
                print("\nYour turn")
                valid_move = False
                while not valid_move:
                    try:
                        row = int(input("Enter row (0-2): "))
                        col = int(input("Enter column (0-2): "))
                        if (row, col) in state.get_legal_moves():
                            state = state.make_move((row, col))
                            valid_move = True
                        else:
                            print("Invalid move, try again.")
                    except ValueError:
                        print("Please enter valid numbers.")
                print(state)
                ai_move = mcts_search(state, iterations=1000)
                state.moves.append([state.symbols.get(state.current_player), ai_move])
                state = state.make_move(ai_move)

        # Game over, display result
        print(state)
        if state.winner == 0:
            print("Game ended in a draw!")
        elif state.winner == 1:
            print("AI wins!")
        else:
            print("You win!")


if __name__ == "__main__":
    play_game()
