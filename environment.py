# environment.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import random

from parking_dfs import ParkingDfs
from GameState import GameState


# !/usr/bin/python

class Environment:

    def getCurrentState(self):
        """
        Returns the current state of enviornment
        """
        abstract

    def getPossibleActions(self, state):
        """
          Returns possible actions the agent
          can take in the given state. Can
          return the empty list if we are in
          a terminal state.
        """
        abstract

    def doAction(self, action):
        """
          Performs the given action in the current
          environment state and updates the enviornment.

          Returns a (reward, nextState) pair
        """
        abstract

    def reset(self):
        """
          Resets the current state to the start state
        """
        abstract

    def isTerminal(self):
        """
          Has the enviornment entered a terminal
          state? This means there are no successors
        """
        state = self.getCurrentState()
        actions = self.getPossibleActions(state)
        return len(actions) == 0


class MdpEnvironment(Environment):

    def __init__(self, parking_mdp, start_location):
        self.parking_mdp = parking_mdp
        self.start_state = GameState(start_location[0], start_location[1], False, False, False, False)
        self.reset()

    def getCurrentState(self):
        return self.state

    def getPossibleActions(self, state):
        return self.parking_mdp.getPossibleActions(state)

    def doAction(self, action):
        successors = self.parking_mdp.getTransitionStatesAndProbs(self.state, action)
        sum = 0.0
        rand = random.random()
        state = self.getCurrentState()
        for nextState, prob in successors:
            sum += prob
            if rand < sum:
                reward = self.parking_mdp.getReward(state, action, nextState)
                self.state = nextState
                return (nextState, reward)
        raise 'Total transition probability less than one; sample failure.'

    def reset(self):
        self.state = self.start_state


class DfsEnvironment(Environment):
    def __init__(self, dfs_agent: ParkingDfs, start_location):
        self.dfs_agent = dfs_agent
        self.start_state = GameState(start_location[0], start_location[1], False, False, False, False)
        self.reset()

    def getCurrentState(self):
        return self.state

    def getPossibleActions(self, state):
        return self.dfs_agent.getPossibleActions(state)

    def doAction(self, action):
        self.state = self.dfs_agent.get_successor(self.state, action)
        reward = self.dfs_agent.getReward(self.state)
        return self.state, reward

    def reset(self):
        self.state = self.start_state
