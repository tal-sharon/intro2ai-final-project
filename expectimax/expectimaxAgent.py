import random

import util
import numpy as np

from agents import Action
from game_state import GameState

SCORE = 0
ACTION = 1


class ExpectimaxAgent(object):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (parking_map.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=5):
        # self.evaluation_function = util.lookup(evaluation_function, globals())
        self.evaluation_function = scoreEvaluationFunction
        self.depth = depth

    def get_action(self, game_state: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.get_action_helper(game_state, 0, 0)[ACTION]

    def get_action_helper(self, cur_state: GameState, cur_depth: int, agent_index: int) -> (int, Action):
        if cur_depth == self.depth or cur_state.done:
            return self.evaluation_function(cur_state), None

        actions = cur_state.get_legal_actions(agent_index)
        successors = [cur_state.generate_successor(action=action, agent_index=agent_index) for action in actions]

        if agent_index == 0:  # Our agent (maximizing)
            successors_scores = [self.get_action_helper(successor, cur_depth + 1, 1)[SCORE] for successor
                                 in successors]
            return max(successors_scores), actions[np.argmax(successors_scores)]

        # Opponent agent (minimizing)
        successors_scores = [self.get_action_helper(successor, cur_depth, 0)[SCORE] for successor
                             in successors]
        probabilities = [action[4] for action in actions]
        return sum([p * s for p, s in zip(probabilities, successors_scores)]), None


def scoreEvaluationFunction(state):
    return - 10 * (
                abs(state.cur_row - state.home_location[0]) + abs(state.cur_col - state.home_location[1])) + state.score
