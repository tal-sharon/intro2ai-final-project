import random

import numpy as np
from matplotlib import pyplot as plt

import graphicsUtils
from Grid import makeGrid
from agents import Action
from display import Display
from environment import MdpEnvironment
from parking_mdp import ParkingMdp
from run_game import runEpisodeWithDisplay, choose_parkings, printString, base_map, close_to_home, small_map, \
    scenario_3_prob, scenario_3_det, scenario_5_prob_05, scenario_4, generate_map, get_board_from_state, scenario_2


def run_episode_train_q_learning(agent, environment, decision, discount, episode, algo, counter=None):
    returns = 0
    totalDiscount = 1
    environment.reset()
    if 'startEpisode' in dir(agent): agent.startEpisode()
    prev_state = None
    while True:

        state = environment.getCurrentState()

        # END IF IN A TERMINAL STATE
        actions = environment.getPossibleActions(state)
        if not actions:
            if episode % 1000 == 0:
                if counter is not None:
                    counter[(prev_state.row, prev_state.col)] += 1
            return returns

        # GET ACTION (USUALLY FROM AGENT)
        action = decision(state)
        if action == None:
            raise 'Error: Agent returned None action'

        # EXECUTE ACTION
        nextState, reward = environment.doAction(action)

        # UPDATE LEARNER
        if 'observeTransition' in dir(agent):
            agent.observeTransition(state, action, nextState, reward)

        returns += reward * totalDiscount
        totalDiscount *= discount
        prev_state = state


def plot_q_learning_training(rewards):
    # Sample data
    y = [sum(rewards[i:i + 10]) / 10 for i in range(0, len(rewards), 10)]
    x = np.linspace(1, len(rewards) + 1, len(y))  # x-axis values

    # Create a single plot
    plt.figure(figsize=(8, 5))  # Set the figure size

    # Plot the single line
    plt.plot(x, y, label='Q-Learning Reward', color='b')  # Customize line with color 'b' (blue)

    # Set titles and labels
    plt.title('Q-Learning reward during training')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()


if __name__ == '__main__':
    parking_map = scenario_4
    det_parking_map = scenario_4
    random.seed(2)
    discount = 0.99
    episodes = 10000

    ###########################
    # GET THE ParkingWorld
    ###########################
    mdp = ParkingMdp(parking_map, (9, 0))
    env = MdpEnvironment(mdp, (0, 19))

    test_mdp = ParkingMdp(parking_map, (9, 0))
    test_env = MdpEnvironment(test_mdp, (0, 19))

    ###########################
    # GET THE AGENT
    ###########################

    import qlearningAgents

    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': 0.99,
                  'alpha': 0.5,
                  'epsilon': 0.3,
                  'actionFn': actionFn,
                  'numTraining': 1000}
    a = qlearningAgents.QLearningAgent(**qLearnOpts)

    ###########################
    # RUN EPISODES
    ###########################

    messageCallback = lambda x: printString(x)
    pauseCallback = lambda: None
    decisionCallback = a.getAction

    # RUN EPISODES
    if episodes > 0:
        print()
        print("RUNNING", episodes, "EPISODES")
        print()
    results_list = []

    for episode in range(1, episodes + 1):
        val = run_episode_train_q_learning(a, env, decisionCallback, discount, episode, "q-learning")
        results_list.append(val)

    display = Display(makeGrid(parking_map), max_val=0.2)
    display.setup()

    a.setEpsilon(0)
    a.setLearningRate(0)
    test_returns = 0
    for episode in range(1, episodes + 1):
        test_returns += runEpisodeWithDisplay(a, test_env, discount, decisionCallback, display, messageCallback,
                                              pauseCallback, episode, parking_map)
    if episodes > 0:
        print()
        print("AVERAGE RETURNS FROM START STATE: " + str((test_returns + 0.0) / episodes))
        print()
        print()
    plot_q_learning_training(results_list)

    input("Press Enter to exit ...")
