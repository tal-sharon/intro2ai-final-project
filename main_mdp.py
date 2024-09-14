import random

import learningAgents
from Grid import makeGrid
from display import Display
from environment import MdpEnvironment
from parking_mdp import ParkingMdp
from run_game import runEpisodeWithDisplay, choose_parkings, printString, base_map, scenario_1, scenario_5_prob_05, scenario_5_prob_01, scenario_5_low_det, scenario_4, scenario_2, scenario_3_prob

if __name__ == '__main__':
    train_parking_map = scenario_4
    test_parking_map = scenario_4
    random.seed(0)
    discount = 0.99
    iters = 50
    episodes = 100

    ###########################
    # GET THE ParkingWorld
    ###########################
    mdp = ParkingMdp(train_parking_map, (9, 0))
    env = MdpEnvironment(mdp, (0, 19))

    ###########################
    # GET THE AGENT
    ###########################

    import valueIterationAgents

    a = valueIterationAgents.ValueIterationAgent(mdp, discount, iters)

    ###########################
    # RUN EPISODES
    ###########################
    display = Display(makeGrid(test_parking_map), max_val=0.15)
    display.setup()

    messageCallback = lambda x: printString(x)
    pauseCallback = lambda: None
    decisionCallback = a.getAction

    # RUN EPISODES
    if episodes > 0:
        print()
        print("RUNNING", episodes, "EPISODES")
        print()
    returns = 0
    test_mdp = ParkingMdp(test_parking_map, (9, 0))
    test_env = MdpEnvironment(test_mdp, (0, 19))
    for episode in range(1, episodes + 1):
        returns += runEpisodeWithDisplay(a, test_env, discount, decisionCallback, display, messageCallback, pauseCallback,
                                         episode, test_parking_map)
    if episodes > 0:
        print()
        print("AVERAGE RETURNS FROM START STATE: " + str((returns + 0.0) / episodes))
        print()
        print()

    input("Press Enter to exit ...")
