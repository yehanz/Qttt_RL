### Run

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

### File Description:

- GameTree.py: Function as the state value function for the TD(0) algorithm
- env.py: Implement the Qttt environment, which provides APIs like get_state, get_valid_moves, step, etc.
- human_agent.py: Interfact for human player
- rl_agent: TD(0) RL agent class and the program driver class which responsible for rl agent training.
- Unit_test: unit tests for the game environment class
