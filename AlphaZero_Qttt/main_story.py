from copy import deepcopy

from AlphaZero_Qttt.MCTS import MCTS
from AlphaZero_Qttt.Network import Network
from AlphaZero_Qttt.env_bridge import EnvForDeepRL
from AlphaZero_Qttt.env_bridge import INDEX_TO_MOVE

def learn_from_self_play(nnet: Network, config):
    """
    Main story for the deep RL agent

    :EnvForDeepRL game_env: game_env environment which provides APIs to interact with the game_env environment
    :param nnet: neural network generate state-value v(s) and probabilistic policy P(a|s)
    :param config: configurable program parameters
        compete_rounds: rounds of games for old/new network competition
        fresh_data_percentage: for each round of data generating, it specifies the portion of data
                    we need to generate in this round, as opposed to stale data we used for training
                    during the last iteration
        max_iteration: max number of network iteration, for each iteration, we train a new
                    network based on the old one, competing with the old one and take the superior
        training_dataset_limit: number of most recent data that would be used for network training, in
                    AlphaZero paper, it uses the most recent 500,000 data points for training
        min_win_rate_network_update: minimum win rate the new network has to achieve over the old
                    one in order to take over the old network
    :return:
    """
    curr_net = nnet

    # store training data generated during self-play
    training_example = []

    # for each iteration, we train a new nn and compete with the older one
    for i in range(config.numMCTSSims):
        print('epoch %d' % i)
        # for each round, keep 50% stale training data from last round
        if len(training_example) > config.training_dataset_limit:
            training_example = training_example[:-int(
                config.training_dataset_limit * (1 - config.fresh_data_percentage))]

        # generate 50% new data with current nn
        while len(training_example) < config.training_dataset_limit:
            training_example += run_one_episode(curr_net, config)

        # training a new nn based on curr nn
        competitor_net = deepcopy(curr_net)
        competitor_net.train(training_example)

        # if competitor_net better than self.curr_net
        new_wins, old_wins, tie = battle(competitor_net, curr_net, config)
        print('win rate: %.3f' % (new_wins / (new_wins + old_wins)))
        if new_wins / (new_wins + old_wins + tie) > config.updateThreshold:
            print('-------------save the better network-----------------')
            # save current network
            competitor_net.save(config)
            # use new network to generate training data
            curr_net = competitor_net


def run_one_episode(curr_net, config):
    game_env = EnvForDeepRL()
    # we can use only 1 tree here
    monte_carlo_trees = [MCTS(deepcopy(game_env), curr_net,
                              config.numMCTSSims, config.cpuct),
                         MCTS(deepcopy(game_env), curr_net,
                              config.numMCTSSims, config.cpuct), ]
    training_examples = []

    while True:
        temp = 1 if game_env.round_ctr < 7 else 0
        # get player's search tree
        mc = monte_carlo_trees[game_env.current_player_id]

        # [states_from_even_piece_view, probabilistic_policy, 1/-1]
        states, policy_given_state = mc.get_action_prob(temp)

        action_code = game_env.pick_a_valid_move(policy_given_state)

        # register data
        training_examples.append([states, policy_given_state,
                                  game_env.current_player_id])

        print(INDEX_TO_MOVE[action_code % 36] if action_code<72 else 'done')
        print(states[0].to_hashable())
        print(states[1].to_hashable())
        print('\n')
        # step action
        _, _, reward, done = game_env.act(action_code)
        for mct in monte_carlo_trees:
            mct.step(action_code)

        if done:
            training_examples = update_reward(training_examples, reward, game_env.current_player_id)
            break

    return training_examples


def update_reward(training_examples, reward, curr_player_id):
    return [(x[0], x[1], reward * (-1) ** (curr_player_id != x[2])) for x in training_examples]


def battle(net1, net2, config):
    """
        compete_rounds: rounds of games for old/new network competition
        max_round_of_simulation: MCTS related parameters, # of rounds of planning before yielding
                    the final action
    """
    score_board = {net1: 0, net2: 0, None: 0}

    for _ in range(config.roundsOfBattle // 2):
        score_board[one_round_of_battle(net1, net2, config)] += 1
    for _ in range(config.roundsOfBattle // 2):
        score_board[one_round_of_battle(net2, net1, config)] += 1

    return score_board[net1], score_board[net2], score_board[None]


def one_round_of_battle(net1, net2, config):
    game_env = EnvForDeepRL()
    mc1 = MCTS(EnvForDeepRL(), net1, config.numMCTSSims, config.cpuct)
    mc2 = MCTS(EnvForDeepRL(), net2, config.numMCTSSims, config.cpuct)
    monte_carlo_trees = [mc1, mc2]

    while True:
        mc = monte_carlo_trees[game_env.current_player_id]

        # set temperature to 0 to get the strongest possible move
        _, policy_given_state = mc.get_action_prob(temp=0)
        action_code = game_env.pick_a_valid_move(policy_given_state, is_train=False)

        # step action
        _, _, reward, done = game_env.act(action_code)
        for mct in monte_carlo_trees:
            mct.step(action_code, is_train=False)

        if done:
            if reward > 0:
                return monte_carlo_trees[game_env.current_player_id].nn
            elif reward < 0:
                return monte_carlo_trees[game_env.current_player_id ^ 1].nn
            else:
                return None
