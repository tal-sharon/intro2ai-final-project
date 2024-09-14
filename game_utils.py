from collections import deque

import numpy as np
from matplotlib import pyplot as plt


def dist_from_home(grid, start):
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    # Initialize distance grid with -1
    distance = [[-1 for _ in range(cols)] for _ in range(rows)]

    queue = deque([(start, 0)])  # queue stores tuples of (position, distance)
    visited = set(start)

    while queue:
        (r, c), dist = queue.popleft()
        distance[r][c] = dist

        if is_parking(grid, r, c):
            continue  # Stop searching this path if a goal state is reached

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] not in ['#', 'e'] and (nr, nc) not in visited:
                queue.append(((nr, nc), dist + 1))
                visited.add((nr, nc))
    return distance


def is_terminal(state):
    return isinstance(state, str)


def is_available_parking(board, x, y):
    """
    Only the TERMINAL_STATE state is *actually* a terminal state.
    The other "exit" states are technically non-terminals with
    a single action "exit" which leads to the true terminal state.
    This convention is to make the grids line up with the examples
    in the R+N textbook.
    """
    return isinstance(board[x][y], float) or board[x][y] == 1


def is_parking(board, row, col):
    return (0 <= row < len(board) and 0 <= col < len(board[0])
            and (isinstance(board[row][col], float) or board[row][col] == 1))


def is_road(board, row, col):
    return (0 <= row < len(board) and 0 <= col < len(board[0])
            and (board[row][col] in (' ', 'c')) and (row, col))


class Reward:
    def __init__(self, board, home_loc):
        self.board = board
        self.dist = dist_from_home(board, home_loc)
        self.max_dist = max(max(self.dist))

    def get_reward(self, state):
        """
        Get reward for state, action, nextState transition.

        Note that the reward depends only on the state being
        departed (as in the R+N book examples, which more or
        less use this convention).
        """
        if is_terminal(state):
            return 0
        row = state.row
        col = state.col
        if is_available_parking(self.board, row, col):
            return self.max_dist - self.dist[row][col]
        # return -0.1 - (self.dist[row][col]/self.max_dist)
        return -0.1


def cal_evg_dist(data):
    matrix = [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 20, -1, -1, -1, -1, 25, -1, -1, -1, -1, 30],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 19, -1, -1, -1, -1, 24, -1, -1, -1, -1, 29],
        [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
        [28, 27, 26, 25, 24, 23, 22, 21, 18, 17, 18, 21, 22, 23, 22, 23, 26, 27, 28, 27],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 16, -1, -1, -1, -1, 21, -1, -1, -1, -1, 26],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 15, -1, -1, -1, -1, 20, -1, -1, -1, -1, 25],
        [23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        [24, 23, 22, 21, 20, 19, 18, 17, 14, 13, 14, 17, 18, 19, 18, 19, 22, 23, 24, 23],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 12, -1, -1, -1, -1, 17, -1, -1, -1, -1, 22],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, 11, -1, -1, -1, -1, 16, -1, -1, -1, -1, 21],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    ]

    # Calculate the weighted average based on the provided data and matrix
    total_sum = 0
    total_count = 0

    for (x, y), count in data.items():
        # Adjust the indexing because matrices are usually indexed by (row, column)
        value = matrix[x][y]
        total_sum += value * count
        total_count += count

    # Calculate the average
    average = total_sum / total_count

    # Print the result
    print(f"Weighted Average: {average}")
    return average


def draw_scenario_6_graph(data):
    # Sample data
    x = np.linspace(1, 100, 100)  # x-axis values

    # Create 4 separate plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))  # 2x2 grid of plots

    titles = ["Reward average", "Reward STD", "Proximity To Home", "Average Travel Distance"]
    labels = ["DFS", "MDP", "Q-Learning"]

    for i in range(4):
        ax = axs[i]  # Select the subplot

        # Generate 3 different lines for each subplot
        for j in range(3):
            y = data[3 * i + j]
            ax.plot(x, y, label=labels[j])

        if i == 3:
            ax.set_ylim([0, 100])

        # Set titles and labels
        ax.set_title(titles[i])
        ax.set_xlabel("maps")
        ax.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
