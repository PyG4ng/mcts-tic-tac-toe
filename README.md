# Monte Carlo Tree Search (MCTS) for Tic-Tac-Toe 
[![Run Tests](https://github.com/PyG4ng/mcts-tic-tac-toe/actions/workflows/tests.yml/badge.svg)](https://github.com/PyG4ng/mcts-tic-tac-toe/actions/workflows/tests.yml)&ensp; 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight implementation of Monte Carlo Tree Search (MCTS) for a Tic-Tac-Toe game.

MCTS is a heuristic search algorithm used in decision-making and game AI. It simulates multiple potential moves and selects the most promising one based on exploration and exploitation strategies.  

## Features

- **Transposition table** – Stores previously visited states to optimize search.
- **Smart rollout** – Enhances move selection efficiency during simulations.


## Requirements

- Python 3.11+


## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/PyG4ng/mcts-tic-tac-toe.git
   cd mcts-tic-tac-toe
   ```

2. Create a virtual environment and install packages:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the game:
   ```bash
   python game_play.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
