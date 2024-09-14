# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import mdp
import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        self.policy = {}
        self.q_values = util.Counter()

        state_reward = util.Counter()
        q_state_reward = util.Counter()
        for s in self.mdp.getStates():
            state_reward[s] = self.mdp.getReward(s, None, None)
            for a in self.mdp.getPossibleActions(s):
                q_state_reward[(s, a)] += self.mdp.getReward(s, a, None)

        for k in range(self.iterations):
            best_next_state = util.Counter()
            for s in self.mdp.getStates():
                next_steps = [sum(p * self.values[s_next] for s_next, p in
                                  mdp.getTransitionStatesAndProbs(s, a)) for a in
                              self.mdp.getPossibleActions(s)]
                best_next_state[s] = max(next_steps) if next_steps else 0
            best_next_state.divideAll(1 / self.discount)
            self.values = (state_reward + best_next_state)

        for s in self.mdp.getStates():
            action_values = util.Counter()
            for a in self.mdp.getPossibleActions(s):
                action_values[a] = sum(p * self.values[s_next] for s_next, p in mdp.getTransitionStatesAndProbs(s, a))
            self.policy[s] = action_values.argMax()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getPolicy(self, state):
        """
          The policy is the best action in the given state
          according to the values computed by value iteration.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        return self.policy[state]

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
