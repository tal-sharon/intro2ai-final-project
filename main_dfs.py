import random

from Grid import makeGrid
from display import Display
from environment import DfsEnvironment, MdpEnvironment
from parking_dfs import ParkingDfs
from run_game import base_map, choose_parkings, printString, runEpisodeWithDisplay, scenario_1, scenario_5_prob_05, scenario_2, scenario_4, scenario_3_prob, scenario_3_det, scenario_5_1_parking, scenario_5_prob_01, scenario_5_low_det

if __name__ == '__main__':
    parking_map = scenario_4
    discount = 1
    iters = 50
    episodes = 100
    home_loc = (9, 0)
    start_loc = (0, 19)

    messageCallback = lambda x: printString(x)
    pauseCallback = lambda: None

    ###########################
    # RUN EPISODES
    ###########################

    # RUN EPISODES
    if episodes > 0:
        print()
        print("RUNNING", episodes, "EPISODES")
        print()
    returns = 0
    display = Display(makeGrid(parking_map))
    display.setup()
    for episode in range(1, episodes + 1):
        ###########################
        # GET THE ParkingWorld
        ###########################
        dfs = ParkingDfs(parking_map, home_loc)
        env = MdpEnvironment(dfs, start_loc)
        decisionCallback = dfs.get_action

        # uncomment to use deterministic version
        # det_parking_map = choose_parkings(parking_map)
        # env = DfsEnvironment(dfs, start_loc)

        returns += runEpisodeWithDisplay(dfs, env, discount, decisionCallback, display, messageCallback, pauseCallback,
                                         episode, parking_map)
    if episodes > 0:
        print()
        print("AVERAGE RETURNS FROM START STATE: " + str((returns + 0.0) / episodes))
        print()
        print()

    input("Press Enter to exit ...")
