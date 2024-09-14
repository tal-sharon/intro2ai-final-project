import random

import numpy as np

import game_utils
import graphicsUtils
import qlearningAgents
import util
from Grid import makeGrid
from display import Display
from environment import MdpEnvironment
from main_q import run_episode_train_q_learning
from parking_dfs import ParkingDfs
from parking_mdp import ParkingMdp
from run_game import base_map, choose_parkings, scenario_1, scenario_2, scenario_3_prob, scenario_3_det, \
    scenario_5_prob_05, generate_map, scenario_5_prob_01, scenario_4
import valueIterationAgents


def run_episode_wo_display(agent, environment, decision, discount, episode, algo, dist_counter, steps_counter):
    returns = 0
    totalDiscount = 1
    environment.reset()
    if 'startEpisode' in dir(agent): agent.startEpisode()
    final_state = None
    while steps_counter[episode] < 1000:

        state = environment.getCurrentState()

        # END IF IN A TERMINAL STATE
        actions = environment.getPossibleActions(state)
        if not actions:
            dist_counter[(final_state.row, final_state.col)] += 1
            return returns

        # GET ACTION (USUALLY FROM AGENT)
        action = decision(state)
        if action == None:
            raise 'Error: Agent returned None action'

        # EXECUTE ACTION
        mdp_nextState, mdp_reward = environment.doAction(action)
        steps_counter[episode] += 1

        returns += mdp_reward * totalDiscount
        totalDiscount *= discount
        final_state = state
    return np.inf


def run_main(train_map, test_map):
    episodes = 100
    random.seed(2)
    home_loc = (9, 0)
    start_loc = (0, 19)
    discount = .99

    # init MDP
    iters = 50
    train_mdp = ParkingMdp(train_map, home_loc)
    test_mdp = ParkingMdp(test_map, home_loc)
    train_mdp_env = MdpEnvironment(train_mdp, start_loc)
    test_mdp_env = MdpEnvironment(test_mdp, start_loc)
    value_iteration_agent = valueIterationAgents.ValueIterationAgent(train_mdp, discount, iters)
    mdp_decision_callback = value_iteration_agent.getAction

    # init DFS
    dfs = ParkingDfs(test_map, home_loc)
    dfs_env = MdpEnvironment(dfs, start_loc)
    dfs_decision_callback = dfs.get_action

    # init Q-learning
    actionFn = lambda state: train_mdp.getPossibleActions(state)
    numTraining = 10000
    qLearnOpts = {'gamma': discount,
                  'alpha': .5,
                  'epsilon': .3,
                  'actionFn': actionFn,
                  'numTraining': numTraining}
    q_learning_agent = qlearningAgents.QLearningAgent(**qLearnOpts)
    q_decision_callback = q_learning_agent.getAction
    q_learning_train_counter = util.Counter()
    for episode in range(numTraining):
        run_episode_train_q_learning(q_learning_agent, train_mdp_env, q_decision_callback, discount,
                                     episode, "q-learning", q_learning_train_counter)
    q_learning_agent.setEpsilon(0)
    q_learning_agent.setLearningRate(0)

    print()
    print("RUNNING", episodes, "EPISODES")
    print()

    # init data for print
    mdp_returns = []
    dfs_returns = []
    q_learning_returns = []
    q_learning_fails = 0
    mdp_final_states = util.Counter()
    dfs_final_states = util.Counter()
    q_learning_final_states = util.Counter()
    mdp_steps_counter = util.Counter()
    dfs_steps_counter = util.Counter()
    q_learning_steps_counter = util.Counter()

    for e in range(1, episodes + 1):
        mdp_returns += [
            run_episode_wo_display(value_iteration_agent, test_mdp_env, mdp_decision_callback, discount, e, "MDP",
                                   mdp_final_states, mdp_steps_counter)]
        dfs_returns += [
            run_episode_wo_display(dfs, dfs_env, dfs_decision_callback, discount, e, "DFS", dfs_final_states,
                                   dfs_steps_counter)]
        res = run_episode_wo_display(q_learning_agent, test_mdp_env, q_decision_callback, discount, e, "Q-LEARNING",
                                     q_learning_final_states, q_learning_steps_counter)
        if res == np.inf:
            q_learning_fails += 1
        else:
            q_learning_returns += [res]
        dfs.reset()

    # uncomment for scenario 6 - print data for the graph
    # scenario_6_statistics(dfs_final_states, dfs_returns, dfs_steps_counter, mdp_final_states, mdp_returns,
    #                       mdp_steps_counter, q_learning_final_states, q_learning_returns, q_learning_steps_counter)

    print_results(dfs_final_states, dfs_returns, dfs_steps_counter, mdp_final_states, mdp_returns, mdp_steps_counter,
                  q_learning_fails, q_learning_final_states, q_learning_returns, q_learning_steps_counter)
    return


def scenario_6_statistics(dfs_final_states, dfs_returns, dfs_steps_counter, mdp_final_states, mdp_returns,
                          mdp_steps_counter, q_learning_final_states, q_learning_returns, q_learning_steps_counter):
    mdp_rewards.append(round(np.average(mdp_returns), 2))
    mdp_std.append(round(np.std(mdp_returns), 2))
    mdp_dist.append(game_utils.cal_evg_dist(mdp_final_states))
    mdp_steps.append(mdp_steps_counter.totalCount() / len(mdp_steps_counter))
    dfs_rewards.append(round(np.average(dfs_returns), 2))
    dfs_std.append(round(np.std(dfs_returns), 2))
    dfs_dist.append(game_utils.cal_evg_dist(dfs_final_states))
    dfs_steps.append(dfs_steps_counter.totalCount() / len(dfs_steps_counter))
    global ql_fails
    if q_learning_final_states.totalCount() > 0:
        ql_rewards.append(round(np.average(q_learning_returns), 2))
        ql_std.append(round(np.std(q_learning_returns), 2))
        ql_dist.append(game_utils.cal_evg_dist(q_learning_final_states))
        ql_steps.append(q_learning_steps_counter.totalCount() / len(q_learning_steps_counter))
    else:
        ql_rewards.append(None)
        ql_std.append(None)
        ql_dist.append(None)
        ql_steps.append(None)
        ql_fails += 1
    print("mdp_rewards: ", mdp_rewards)
    print("mdp_std: ", mdp_std)
    print("mdp_dist: ", mdp_dist)
    print("mdp_steps: ", mdp_steps)
    print("dfs_rewards: ", dfs_rewards)
    print("dfs_std: ", dfs_std)
    print("dfs_dist: ", dfs_dist)
    print("dfs_steps: ", dfs_steps)
    print("ql_rewards: ", ql_rewards)
    print("ql_std: ", ql_std)
    print("ql_dist: ", ql_dist)
    print("ql_steps: ", ql_steps)
    print("ql_fails: ", ql_fails)


def print_results(dfs_final_states, dfs_returns, dfs_steps_counter, mdp_final_states, mdp_returns, mdp_steps_counter,
                  q_learning_fails, q_learning_final_states, q_learning_returns, q_learning_steps_counter):
    print("AVERAGE RETURNS FROM START STATE - MDP: " + str(round(np.average(mdp_returns), 2)))
    print("STD RETURNS FROM START STATE - MDP: " + str(round(np.std(mdp_returns), 2)))
    print()
    print("MDP final_states: ")
    print(game_utils.cal_evg_dist(mdp_final_states))
    print(mdp_final_states)
    print("MDP steps_counter: ")
    print(mdp_steps_counter.totalCount() / len(mdp_steps_counter))
    print()
    print("AVERAGE RETURNS FROM START STATE - DFS: " + str(round(np.average(dfs_returns), 2)))
    print("STD RETURNS FROM START STATE - DFS: " + str(round(np.std(dfs_returns), 2)))
    print()
    print("DFS final_states: ")
    print(dfs_final_states)
    print(game_utils.cal_evg_dist(dfs_final_states))
    print("DFS steps_counter: ")
    print(dfs_steps_counter.totalCount() / len(dfs_steps_counter))
    print()
    print("AVERAGE RETURNS FROM START STATE - Q-LEARNING: " + str(round(np.average(q_learning_returns), 2)))
    print("STD RETURNS FROM START STATE - Q-LEARNING: " + str(round(np.std(q_learning_returns), 2)))
    print()
    print()
    print("QL final_states: ")
    print(q_learning_final_states)
    if q_learning_final_states.totalCount() > 0:
        print(game_utils.cal_evg_dist(q_learning_final_states))
    print("QL steps_counter: ")
    print(q_learning_steps_counter.totalCount() / len(q_learning_steps_counter))
    print("QL fail runs: ")
    print(q_learning_fails)


def display_final_parking_dist(map, parkings_counter: util.Counter):
    print(parkings_counter)
    for row in range(len(map)):
        for col in range(len(map[0])):
            if isinstance(map[row][col], float) or isinstance(map[row][col], int):
                map[row][col] = 1.0 * parkings_counter[(row, col)] / parkings_counter.totalCount()
    display = Display(makeGrid(map))
    display.setup()
    display.drawGrid(map, False, None, False)
    graphicsUtils.sleep(2)


if __name__ == '__main__':
    # scenario 1
    run_main(scenario_1, scenario_1)

    # scenario 2
    # run_main(scenario_2, scenario_2)

    # scenario 3
    # run_main(scenario_3_prob, scenario_3_det)

    # scenario 4
    # run_main(scenario_4, scenario_4)

    # scenario 5
    # run_main(scenario_5_prob_01, scenario_5_prob_01)

    # scenario 6
    mdp_rewards = []
    dfs_rewards = []
    ql_rewards = []
    mdp_dist = []
    dfs_dist = []
    ql_dist = []
    mdp_steps = []
    dfs_steps = []
    ql_steps = []
    mdp_std = []
    dfs_std = []
    ql_std = []
    ql_fails = 0

    # still scenario 6
    for i in range(1):
        print("map ", i)
        parking_map = generate_map(base_map)
        run_main(parking_map, parking_map)

    rewards_avg = [dfs_rewards, mdp_rewards, ql_rewards]
    rewards_std = [dfs_std, mdp_std, ql_std]
    dist = [dfs_dist, mdp_dist, ql_dist]
    steps = [dfs_steps, mdp_steps, ql_steps]
    try:
        game_utils.draw_scenario_6_graph(rewards_avg + rewards_std + dist + steps)
    except Exception as e:
        print("Uncomment lines 121-122 to display the graph")
