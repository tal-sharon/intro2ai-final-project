## Run simulation with GUI
On each of the following main files we imported the maps of each scenario.
The numbers of the scenarios match the numbers of the scenarios described in the results section in the PDF.
To run the different scenarios you can replace the variables in the files as described below: 
* **DFS** - run main_dfs.py. You can change the value of the variable parking_map to change the map. 
* **MDP** - run main_mdp.py. Change the value of train_parking_map to choose the map the MDP trains on,
    and change test_parking_map to choose the map the MDP is tested on.
    For the presented scenarios we used the same map for both (probability version).
    During the coding process we sometimes used the deterministic map versions as the test map to easier debugging. 
* **Q-learning** - run main_q.py. The map usage is the same as in MDP.

## Comparison between the algorithms 
To run the comparison between the three algorithms run main_compare.py.
 We used it to collect data for the statistics of each of the scenarios described in the PDF.
 To run any of them uncomment the relevant lines. By default, it runs only scenario 1.
 To run scenario 3 for example, comment the line below "# scenario 1" and uncomment the line below "# scenario 3".
 Pay attention, in order to run scenario 6 with the accumulated data you need to uncomment several lines, including in the method run_main (in the same file).