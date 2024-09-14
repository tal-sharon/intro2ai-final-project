# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import numpy as np

from learningAgents import ReinforcementAgent

import random, util


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - getQValue
        - getAction
        - getValue
        - getPolicy
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions
          for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_values = util.Counter()
        self.policy = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we never seen
          a state or (state,action) tuple
        """
        return self.q_values[(state, action)]

    def getValue(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        return max([self.getQValue(state, action) for action in self.getLegalActions(state)]
                   ) if self.getLegalActions(state) else 0

    def getPolicy(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        max_value = -np.inf
        best_actions = []
        for action in self.getLegalActions(state):
            if self.getQValue(state, action) > max_value:
                max_value = self.getQValue(state, action)
                best_actions = [action]
            elif self.getQValue(state, action) == max_value:
                best_actions.append(action)
        return random.choice(best_actions) if best_actions else None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        best_step_value = max([self.getQValue(nextState, nextAction) for nextAction in
                               self.getLegalActions(nextState)]) if self.getLegalActions(nextState) else 0
        self.q_values[(state, action)] = self.getQValue(state, action) + self.alpha * (
                reward + (self.discount * best_step_value) - self.getQValue(state, action))


