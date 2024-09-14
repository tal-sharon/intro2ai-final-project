# gridworld.py
# ------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from collections import deque

import mdp

from GameState import GameState
from agents import Action
from game_utils import is_terminal, is_available_parking, Reward, is_parking, is_road


class ParkingMdp(mdp.MarkovDecisionProcess):

    def __init__(self, board, home_loc):
        # layout
        self.board = board
        self.reward = Reward(board, home_loc)
        # parameters
        self.livingReward = 0.0
        self.terminalState = "TERMINAL STATE"

    def setLivingReward(self, reward):
        """
        The (negative) reward for exiting "normal" states.

        Note that in the R+N text, this reward is on entering
        a state and therefore is not clearly part of the state's
        future rewards.
        """
        self.livingReward = reward

    def getPossibleActions(self, state):
        """
        Returns list of valid actions for 'state'.

        Note that you can request moves into walls and
        that "exit" states transition to the terminal
        state under the special action "done".
        """
        if is_terminal(state):
            return []
        if is_available_parking(self.board, state.row, state.col):
            return [Action.EXIT]

        x, y = state.row, state.col
        actions = []
        if self.__is_allowed(x - 1, y, state.u):
            actions.append(Action.UP)
        if self.__is_allowed(x + 1, y, state.d):
            actions.append(Action.DOWN)
        if self.__is_allowed(x, y - 1, state.l):
            actions.append(Action.LEFT)
        if self.__is_allowed(x, y + 1, state.r):
            actions.append(Action.RIGHT)
        return actions

    def getStates(self):
        """
        Return list of all states.
        """
        states = [self.terminalState]
        for x in range(len(self.board)):
            for y in range(len(self.board[0])):
                if self.board[x][y] not in ['#', 'e', 0]:
                    right, left, up, down = [False], [False], [False], [False]
                    if is_parking(self.board, x - 1, y):
                        up.append(True)
                    if is_parking(self.board, x + 1, y):
                        down.append(True)
                    if is_parking(self.board, x, y - 1):
                        left.append(True)
                    if is_parking(self.board, x, y + 1):
                        right.append(True)

                    for l in left:
                        for r in right:
                            for u in up:
                                for d in down:
                                    states.append(GameState(x, y, l, r, u, d))
        return states

    def getReward(self, state, action, nextState):
        """
        Get reward for state, action, nextState transition.

        Note that the reward depends only on the state being
        departed (as in the R+N book examples, which more or
        less use this convention).
        """
        return self.reward.get_reward(state)

    def getStartState(self):
        for x in range(len(self.board)):
            for y in range(len(self.board[0])):
                if self.board[x][y] == ' ':
                    return GameState(x, y, False, False, False, False)
        raise 'Board has no start state'

    def getTransitionStatesAndProbs(self, state, action):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.
        """
        if action not in self.getPossibleActions(state):
            raise "Illegal action!"

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

    def __is_allowed(self, row, col, is_available_parking):
        return is_road(self.board, row, col) or is_available_parking

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
                        actions.append((GameState(row, col, l, r, u, d), round(p_l * p_r * p_u * p_d, 5)))
        return actions
