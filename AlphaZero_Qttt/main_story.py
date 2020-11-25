import numpy as np
from copy import deepcopy
from env import REWARD
from AlphaZero_Qttt.env_bridge import EnvForDeepRL
from AlphaZero_Qttt.Network import Network

def learn_from_self_play(game_env: EnvForDeepRL, nnet:Network, config):
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
    for i in range(config.max_iteration):

        # for each round, keep 50% stale training data from last round
        if len(training_example) > config.training_dataset_limit:
            training_example = training_example[:-int(
                config.upper_limit * (1 - config.fresh_data_percentage))]

        # generate 50% new data with current nn
        while len(training_example) < config.training_dataset_limit:
            training_example += run_one_episode(game_env, curr_net, config)

        # training a new nn based on curr nn
        competitor_net = deepcopy(curr_net)
        competitor_net.train(training_example)

        # if competitor_net better than self.curr_net
        new_wins, old_wins = battle(game_env, competitor_net, curr_net, config)
        if new_wins / (new_wins + old_wins) > config.min_win_rate_network_update:
            # save current network
            save(competitor_net)
            # use new network to generate training data
            curr_net = competitor_net


def run_one_episode(game_env: EnvForDeepRL, curr_net, config):
    # we can use only 1 tree here
    monte_carlo_trees = [MTCS(deepcopy(game_env).change_to_even_pieces_view(),
                              curr_net, config.exploration_level),
                         MTCS(deepcopy(game_env).change_to_even_pieces_view(),
                              curr_net, config.exploration_level), ]
    training_examples = []

    while True:
        # get player's search tree
        mc = monte_carlo_trees[game_env.current_player_id]

        # [state_from_even_piece_view, probabilistic_policy, 1/-1]
        state, policy_given_state = mc.get_action_prob()
        action_code = game_env.pick_a_valid_move(policy_given_state)

        # step action
        _, _, winner, done = game_env.act(action_code)
        for mct in monte_carlo_trees:
            mct.step(action_code)

        # register data
        training_examples.append([state, policy_given_state,
                                  (-1) ** game_env.current_player_id])

        if done:
            reward = REWARD[winner + '_REWARD']
            update_reward(training_examples, reward)
            break

    return training_examples


def update_reward(training_examples, reward):
    for example in training_examples:
        example[-1] *= reward


def battle(game_env, competitor_net, curr_net, config):
    """
        compete_rounds: rounds of games for old/new network competition
        max_round_of_simulation: MCTS related parameters, # of rounds of planning before yielding
                    the final action
    """
    # we can use only 1 tree here
    competitor_mc = MTCS(deepcopy(game_env).change_to_even_pieces_view(),
                         competitor_net, config.exploration_level)
    curr_mc = MTCS(deepcopy(game_env).change_to_even_pieces_view(),
                   curr_net, config.exploration_level)
    monte_carlo_trees = [competitor_mc, curr_mc]

    score_board = {competitor_mc: 0, curr_mc: 0}

    for _ in range(config.compete_rounds):
        while True:
            mc = monte_carlo_trees[game_env.current_player_id]

            # [state_from_even_piece_view, probabilistic_policy, None]
            state, policy_given_state = mc.get_action_prob()
            action_code = game_env.pick_a_valid_move(policy_given_state)

            # step action
            _, _, winner, done = game_env.act(action_code)
            for mct in monte_carlo_trees:
                mct.step(action_code)

            if done:
                reward = REWARD[winner + '_REWARD']
                if reward > 0:
                    score_board[monte_carlo_trees[game_env.current_player_id]] += 1
                elif reward < 0:
                    score_board[monte_carlo_trees[game_env.current_player_id ^ 1]] += 1
                # swap player sequence to make a different play take the first hand in the next round
                monte_carlo_trees.reverse()
                break

        return score_board[competitor_mc], score_board[curr_mc]


def save(competitor_net, *args):
    pass
