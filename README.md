### Run the TD(0) agent

```shell
# Learn by playing with itself
python rl_agent.py auto [options]

Options:
	-s, --save	string	Filename to save the model, mandatory
	-l, --load	string	Filename to load the model, optional
	
# Learn by playing with human
python rl_agent.py human [options]

Options:
	-s, --save	string	Filename to save the model, mandatory
	-l, --load	string	Filename to load the model, optional
```
### Play with the AlphaZero
AlphaZero_Qttt/play_with_human.py: Download the AlphaZero network checkpoint, configure the checkpoint path so that it can be loaded by the python script. Run this file to play with the agent.

### File Description:

- GameTree.py: Function as the state value function for the TD(0) algorithm
- env.py: Implement the Qttt environment, which provides APIs like get_state, get_valid_moves, step, etc.
- human_agent.py: Interfact for human player
- rl_agent.py: TD(0) RL agent class and the program driver class which responsible for rl agent training.
- Unit_test: unit tests for the game environment class
- AlphaZero_Qttt:
    + driver.py: the driver program for AlphaZero training
    + env_bridge.py: env wrapper which provides compatible APIs for AlphaZero context
    + main_story.py: include all functions needed to train the agent
    + MCTS.py: implement monte carlo search tree
    + Network.py: implement the deep neural network for AlphaZero
    + networkBattle.py: an independent file that used to check which agent is more superior
    + play_with_human.py: an independent file that implements a simple interface for man-machine competition
 