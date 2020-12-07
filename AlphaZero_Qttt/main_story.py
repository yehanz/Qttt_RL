import pickle
from copy import deepcopy

from AlphaZero_Qttt.MCTS import MCTS
from AlphaZero_Qttt.Network import Network
from AlphaZero_Qttt.env_bridge import EnvForDeepRL
from Utils.get_sym import board_transforms, prob_vec_transforms


def learn_from_self_play(nnet: Network, config, training_example=None):
    """
    Main story for the deep RL agent

    :param training_example: store training data generated during self-play
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
    training_example = [] if training_example is None else training_example
    # for each iteration, we train a new nn and compete with the older one
    for epoch in range(config.numIters):
        print('epoch %d' % epoch)

        # keep some stale data
        if not (epoch == 0 and config.skip_initial_data_drop) and \
                len(training_example) > config.training_dataset_limit:
            training_example = training_example[:-int(
                config.training_dataset_limit * (1 - config.fresh_data_percentage))]

        # generate some new data with current nn
        num_data = len(training_example)
        while num_data < config.training_dataset_limit:
            new_data = run_one_episode(curr_net, config)
            num_data += len(new_data)
            training_example += new_data

        # save training examples for checkpoint since they are extremely time-consuming
        # to generate
        pickle.dump(training_example, open(
            config.path_checkpoints + config.training_examples_filename, "wb"))

        # training a new nn based on curr nn
        competitor_net = deepcopy(curr_net)
        competitor_net.train(training_example)

        # if competitor_net better than self.curr_net
        new_wins, old_wins, tie = battle(competitor_net, curr_net, config)
        print('win rate: %.3f, win %d loss %d tie %d' %
              (new_wins / config.roundsOfBattle, new_wins, old_wins, tie))

        if new_wins / config.roundsOfBattle > config.updateThreshold or \
                new_wins + tie > 0.9 * config.roundsOfBattle:
            print('-------------save the better network-----------------')
            # save current network
            competitor_net.save(config)
            # use new network to generate training data
            curr_net = competitor_net
            config.numMCTSSims = 400
            config.training_dataset_limit = 4000*8
        else:
            # increase the policy evaluation power if no improvment is observed this term
            config.numMCTSSims = int(1.2 * config.numMCTSSims)
            config.training_dataset_limit = int(1.2 * config.training_dataset_limit)


def run_one_episode(curr_net, config):
    game_env = EnvForDeepRL()
    mc = MCTS(EnvForDeepRL(), curr_net, config.numMCTSSims, config.cpuct)
    training_examples = []

    while True:
        temp = 1 if game_env.round_ctr > 7 else 0

        # [states_from_even_piece_view, probabilistic_policy, 1/0]
        states, policy_given_state = mc.get_action_prob(temp)

        action_code = game_env.pick_a_valid_move(policy_given_state)

        # register data
        training_examples += record_training_data(
            states, policy_given_state, game_env.current_player_id)

        # step action
        _, _, reward, done = game_env.act(action_code)
        mc.step(action_code)

        if done:
            training_examples = update_reward(training_examples, reward, game_env.current_player_id)
            break

    return training_examples


def record_training_data(states, policy_given_state, curr_player_id):
    training_data = []
    # get all symmetric cases of the chess board
    # ATTENTION: when apply board_transforms[i] to qttt tensor, we must apply
    # prob_vec_tranforms[i] correspondingly to the prob vector!
    s1_tensor, s2_tensor = states[0].to_tensor(), states[1].to_tensor()
    for i in range(len(board_transforms)):
        training_data.append([
            # transformed qttt tensor tuple
            (board_transforms[i](s1_tensor), board_transforms[i](s2_tensor),),
            # transformed qttt probability vector
            policy_given_state[prob_vec_transforms[i]],
            # current player id
            curr_player_id])
    return training_data


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
