from copy import deepcopy
import random

import graphicsUtils
from agents import Action, OpponentAgent
from display import Display
from expectimax.expectimaxAgent import ExpectimaxAgent
from expectimax.game_state import GameState


class Game(object):
    def __init__(self, agent, opponent_agent, display, sleep_between_actions=True):
        super(Game, self).__init__()
        self.sleep_between_actions = sleep_between_actions
        self.agent = agent
        self.display = display
        self.opponent_agent = opponent_agent
        self._state = None
        self._should_quit = False

    def run(self, initial_state):
        self._should_quit = False
        self._state = initial_state
        self.display.setup()
        return self._game_loop()

    def quit(self):
        self._should_quit = True
        self.agent.stop_running()
        self.opponent_agent.stop_running()

    def _game_loop(self):
        while not self._state.done and not self._should_quit:
            if self.sleep_between_actions:
                graphicsUtils.sleep(1)
            self.display.drawGrid(self._state.board)
            print("_____________")
            action = self.agent.getAction(self._state)
            print(action)
            if action == Action.STOP:
                return
            self._state.apply_action(action)
            opponent_action = self.opponent_agent.getAction(self._state)
            self._state.apply_opponent_action(opponent_action)
            self.display.drawGrid(self._state.board)
        return self._state.score


def choose_number(num_parkings):
    numbers = list(range(1, 11))  # Numbers 1 to 10
    probabilities = [0.05, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1]  # Probabilities for each number

    # Choose a number based on the specified probabilities
    chosen_number = random.choices(numbers, probabilities, k=num_parkings)[0]

    return chosen_number


def choose_numbers_from_grid(grid, n):
    # Extract all numbers from the grid
    num_parkings = random.randint(1, n)
    numbers = []
    for row in grid:
        for cell in row:
            if isinstance(cell, (int, float)):
                numbers.append(cell)

    # Normalize the probabilities so they sum to 1
    total = sum(numbers)
    probabilities = [num / total for num in numbers]

    # Choose 'n' numbers based on the extracted probabilities
    chosen_numbers = random.choices(numbers, probabilities, k=num_parkings)

    return chosen_numbers


def choose_parkings(probability_parking_map):
    parking_map = deepcopy(probability_parking_map)
    for i in range(len(parking_map)):
        for j in range(len(parking_map[i])):
            if isinstance(parking_map[i][j], float):
                parking_map[i][j] = 1 if random.random() < parking_map[i][j] else 0
    return parking_map


first_map = [
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' ', '#', '#', '#', '#', 'c'],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [1, 0.12, 0.1, 0.03, ' ', 0.08, 0.01, 0.09, 0.01, ' ', 0.04, 0.1, 0.07, 0.01, ' ', 0.10, 0.05, 0.1, 1, ' '],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' '],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [0.07, 0.02, 0.05, 0.08, ' ', 0.03, 0.09, 0.06, 0.15, ' ', 0.10, 0.12, 0.1, 0.04, ' ', 0.1, 0.06, 0.1, 0.05,
     ' '],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' '],
    ['e', '#', '#', '#', '#', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' ', '#', '#', '#', '#', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [0.1, 0.05, 0.1, 0.07, ' ', 0.10, 0.04, 0.1, 0.06, ' ', 0.09, 0.02, 0.08, 0.15, ' ', 0.07, 0.08, 0.03, 0.1, ' ']
]


def test_actions(board):
    state = GameState((2, 0), board)
    print(state.get_legal_actions(0))
    print(state.get_legal_actions(1))


def test_opponent_agent(parking_map, board):
    state = GameState((2, 0), board)
    agent = OpponentAgent(parking_map)
    print(agent.getAction(state))


def test_move(board, parking_map, display):
    state = GameState((0, 19), board, (9, 0))
    opponent_agent = OpponentAgent(parking_map)
    expectimax_agent = ExpectimaxAgent(display)
    game = Game(expectimax_agent, opponent_agent, display)
    game.run(state)


if __name__ == "__main__":
    random.seed(0)
    probability_map = first_map
    agent = ExpectimaxAgent()
    display = Display(probability_map, 50)
    # game=Game(agent,display)

    display.setup()
    display.drawGrid(probability_map)
    parking_map = choose_parkings(probability_map)
    display.drawGrid(parking_map)
    # test_actions(probability_map)
    # test_opponent_agent(parking_map, probability_map)
    test_move(probability_map, parking_map, display)
    graphicsUtils.sleep(100)
