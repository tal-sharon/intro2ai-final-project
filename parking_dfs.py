import random
from typing import List, Optional

import util
from GameState import GameState
from agents import Agent, Action
from game_utils import is_terminal, is_available_parking, Reward, is_parking, is_road


def reverse_action(action):
    if action == Action.UP:
        return Action.DOWN
    if action == Action.DOWN:
        return Action.UP
    if action == Action.LEFT:
        return Action.RIGHT
    if action == Action.RIGHT:
        return Action.LEFT


class ParkingDfs():
    def __init__(self, board, home_loc):
        self.board = board
        self.reward = Reward(board, home_loc)
        print(self.reward.dist)
        self.visited = set()
        self.previous_actions_stack = util.Stack()
        self.terminalState = "TERMINAL STATE"

    def get_action(self, state: GameState) -> Optional[Action]:
        action = self.getPossibleActions(state)[0]
        if action:
            self.previous_actions_stack.push(action)
            return action
        try:
            return self.go_back()
        except:
            self.reset()
            return self.get_action(state)

    def get_neighbor_parking(self, state):
        possible_actions = []
        if state.u:
            possible_actions.append(Action.UP)
        if state.d:
            possible_actions.append(Action.DOWN)
        if state.r:
            possible_actions.append(Action.RIGHT)
        if state.l:
            possible_actions.append(Action.LEFT)
        return random.choice(possible_actions) if possible_actions else None

    def get_next_road(self, x, y):
        possible_actions = []
        if self.__is_allowed(x - 1, y):
            possible_actions.append(Action.UP)
        if self.__is_allowed(x + 1, y):
            possible_actions.append(Action.DOWN)
        if self.__is_allowed(x, y - 1):
            possible_actions.append(Action.LEFT)
        if self.__is_allowed(x, y + 1):
            possible_actions.append(Action.RIGHT)
        return random.choice(possible_actions) if possible_actions else None

    def getPossibleActions(self, state: GameState) -> List[Action]:
        if is_terminal(state):
            return []
        if is_available_parking(self.board, state.row, state.col):
            return [Action.EXIT]

        action = self.get_neighbor_parking(state)
        if not action:
            action = self.get_next_road(state.row, state.col)
        return [action]

    def get_successor(self, state: GameState, action: Action) -> GameState:
        if action == Action.EXIT:
            return self.terminalState

        row, col = state.row, state.col
        self.visited.add((row, col))

        if action == Action.UP:
            row -= 1
        if action == Action.DOWN:
            row += 1
        if action == Action.LEFT:
            col -= 1
        if action == Action.RIGHT:
            col += 1

        up = True if is_parking(self.board, row - 1, col) else False
        down = True if is_parking(self.board, row + 1, col) else False
        left = True if is_parking(self.board, row, col - 1) else False
        right = True if is_parking(self.board, row, col + 1) else False

        return GameState(row, col, left, right, up, down)

    def go_back(self):
        return reverse_action(self.previous_actions_stack.pop())

    def __is_allowed(self, row, col):
        return is_road(self.board, row, col) and (row, col) not in self.visited

    def getReward(self, state, action, nextState):
        return self.reward.get_reward(state)

    def getTransitionStatesAndProbs(self, state, action):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.
        """
        self.visited.add((state.row, state.col))

        if is_available_parking(self.board, state.row, state.col):
            return [(self.terminalState, 1)]

        if is_terminal(state):
            return []

        row, col = state.row, state.col
        if action == Action.UP:
            row -= 1
        if action == Action.DOWN:
            row += 1
        if action == Action.LEFT:
            col -= 1
        if action == Action.RIGHT:
            col += 1

        return self.get_trans_prob(row, col)

    def get_trans_prob(self, row, col):
        right, left, up, down = [False], [False], [False], [False]
        p_up, p_down, p_left, p_right = 0, 0, 0, 0
        if is_parking(self.board, row - 1, col):
            up.append(True)
            p_up = self.board[row - 1][col]
        if is_parking(self.board, row + 1, col):
            down.append(True)
            p_down = self.board[row + 1][col]
        if is_parking(self.board, row, col - 1):
            left.append(True)
            p_left = self.board[row][col - 1]
        if is_parking(self.board, row, col + 1):
            right.append(True)
            p_right = self.board[row][col + 1]

        actions = []
        for l in left:
            p_l = (l * p_left + (1 - l) * (1 - p_left))
            for r in right:
                p_r = (r * p_right + (1 - r) * (1 - p_right))
                for u in up:
                    p_u = (u * p_up + (1 - u) * (1 - p_up))
                    for d in down:
                        p_d = (d * p_down + (1 - d) * (1 - p_down))
                        actions.append((GameState(row, col, l, r, u, d), p_l * p_r * p_u * p_d))
        return actions

    def reset(self):
        self.visited = set()
        self.previous_actions_stack = util.Stack()
