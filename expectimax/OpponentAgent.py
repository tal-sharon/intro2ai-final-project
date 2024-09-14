from collections import namedtuple

from agents import Agent

OpponentAction = namedtuple('OpponentAction', ['left', 'right', 'up', 'down', 'probability'])


class OpponentAgent(Agent):
    def __init__(self, parking_map):
        super(OpponentAgent, self).__init__()
        self.parking_map = parking_map

    def getAction(self, game_state):
        right = 1 if self.is_parking(game_state.cur_row, game_state.cur_col + 1) else 0
        left = 1 if self.is_parking(game_state.cur_row, game_state.cur_col - 1) else 0
        up = 1 if self.is_parking(game_state.cur_row - 1, game_state.cur_col) else 0
        down = 1 if self.is_parking(game_state.cur_row + 1, game_state.cur_col) else 0
        return OpponentAction(left, right, up, down, 1)

    def is_parking(self, row, col):
        return (0 <= row < len(self.parking_map) and 0 <= col < len(self.parking_map[0])
                and self.parking_map[row][col] == 1)
