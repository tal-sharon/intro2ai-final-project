import copy
from itertools import product

import numpy as np

from agents import Action, OpponentAction

DEFAULT_BOARD_SIZE = 4


class GameState(object):
    def __init__(self, current_location, board, home_location, score=0, done=False):
        super(GameState, self).__init__()
        self._done = done
        self._score = score
        self._board = board
        self._num_of_rows, self._num_of_columns = len(board), len(board[0])
        self.cur_row, self.cur_col = current_location
        self.home_location = home_location

    @property
    def done(self):
        return self._done

    @property
    def score(self):
        return self._score

    @property
    def max_tile(self):
        return np.max(self._board)

    @property
    def board(self):
        return self._board

    def get_legal_actions(self, agent_index):
        if agent_index == 0:
            return self.get_agent_legal_actions()
        elif agent_index == 1:
            return self.get_opponent_legal_actions()
        else:
            raise Exception("illegal agent index.")

    def get_agent_legal_actions(self):
        legal_actions = []
        if self.is_legal(self.cur_row - 1, self.cur_col):
            legal_actions.append(Action.UP)
        if self.is_legal(self.cur_row + 1, self.cur_col):
            legal_actions.append(Action.DOWN)
        if self.is_legal(self.cur_row, self.cur_col - 1):
            legal_actions.append(Action.LEFT)
        if self.is_legal(self.cur_row, self.cur_col + 1):
            legal_actions.append(Action.RIGHT)
        return legal_actions

    # def get_opponent_legal_actions(self):
    #     actions = [OpponentAction(l, r, u, d, )
    #                for l in ([0, 1] if self.is_parking(self.cur_row, self.cur_col - 1) else [0])
    #                for r in ([0, 1] if self.is_parking(self.cur_row, self.cur_col + 1) else [0])
    #                for u in ([0, 1] if self.is_parking(self.cur_row - 1, self.cur_col) else [0])
    #                for d in ([0, 1] if self.is_parking(self.cur_row + 1, self.cur_col) else [0])]
    #
    #     return actions

    def get_opponent_legal_actions(self):
        right, left, up, down = [0], [0], [0], [0]
        p_up, p_down, p_left, p_right = 0, 0, 0, 0
        if self.is_parking(self.cur_row - 1, self.cur_col):
            up.append(1)
            p_up = self.board[self.cur_row - 1][self.cur_col]
        if self.is_parking(self.cur_row + 1, self.cur_col):
            down.append(1)
            p_down = self.board[self.cur_row + 1][self.cur_col]
        if self.is_parking(self.cur_row, self.cur_col - 1):
            left.append(1)
            p_left = self.board[self.cur_row][self.cur_col - 1]
        if self.is_parking(self.cur_row, self.cur_col + 1):
            right.append(1)
            p_right = self.board[self.cur_row][self.cur_col + 1]

        actions = []
        p = 1
        for l in left:
            p *= (l * p_left + (1 - l) * (1 - p_left))
            for r in right:
                p *= (r * p_right + (1 - r) * (1 - p_right))
                for u in up:
                    p *= (u * p_up + (1 - u) * (1 - p_up))
                    for d in down:
                        p *= (d * p_down + (1 - d) * (1 - p_down))
                        actions.append(OpponentAction(l, r, u, d, p))
        return actions

    def is_parking(self, row, col):
        return (0 <= row < len(self.board) and 0 <= col < len(self.board[0])
                and (isinstance(self.board[row][col], float) or self.board[row][col] == 1))

    def is_legal(self, row, col):
        return (0 <= row < len(self.board) and 0 <= col < len(self.board[0])
                and self.board[row][col] not in ("#", "e", 0))

    def apply_opponent_action(self, action):
        if self.is_parking(self.cur_row, self.cur_col - 1):
            self.board[self.cur_row][self.cur_col - 1] = action[0]
        if self.is_parking(self.cur_row, self.cur_col + 1):
            self.board[self.cur_row][self.cur_col + 1] = action[1]
        if self.is_parking(self.cur_row - 1, self.cur_col):
            self.board[self.cur_row - 1][self.cur_col] = action[2]
        if self.is_parking(self.cur_row + 1, self.cur_col):
            self.board[self.cur_row + 1][self.cur_col] = action[3]

    def apply_action(self, action):
        self.board[self.cur_row][self.cur_col] = ' '
        if action == Action.UP:
            self.cur_row -= 1
        if action == Action.DOWN:
            self.cur_row += 1
        if action == Action.LEFT:
            self.cur_col -= 1
        if action == Action.RIGHT:
            self.cur_col += 1
        if self.board[self.cur_row][self.cur_col] == 1:
            self._done = True
            self._score += 10
            # self._score -= (abs(self.cur_row - self.home_location[0]) + abs(self.cur_col - self.home_location[1]))
        self._score -= 0
        self.board[self.cur_row][self.cur_col] = 'c'

    def generate_successor(self, agent_index=0, action=Action.STOP):
        successor = GameState(board=copy.deepcopy(self._board), current_location=(self.cur_row, self.cur_col),
                              score=self.score, done=self._done, home_location=self.home_location)
        if agent_index == 0:
            successor.apply_action(action)
        elif agent_index == 1:
            successor.apply_opponent_action(action)
        else:
            raise Exception("illegal agent index.")
        return successor
