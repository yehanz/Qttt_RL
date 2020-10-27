import numpy as np


def learn_from_self_play(game, nnet, config):
    """
    Main story for the deep RL agent

    :param game: game environment which provides APIs to interact with the game environment
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
            # TODO: use a new MC tree for each episode?
            training_example += run_one_episode(game, curr_net, config)

        # training a new nn based on curr nn
        competitor_net = nnet.deep_copy()
        train(competitor_net, training_example)

        # if competitor_net better than self.curr_net
        new_wins, old_wins = battle(game, competitor_net, curr_net, config)
        if new_wins / (new_wins + old_wins) > config.min_win_rate_network_update:
            # save current network
            save(competitor_net)
            # use new network to generate training data
            curr_net = competitor_net


def encode_action_space(param):
    pass


def run_one_episode(env, curr_net, config):
    # we can use only 1 tree here
    monte_carlo_trees = [MTCS(curr_net, config.exploration_level),
                         MTCS(curr_net, config.exploration_level), ]
    training_examples = []

    while True:
        curr_qttt, whose_turn = env.get_state_from_constant_view()

        mc = monte_carlo_trees[whose_turn]

        for _ in range(config.rounds_of_planning):
            mc.search(curr_qttt)

        # [state, p, None]
        state, policy_given_state = mc.get_policy_for_state(curr_qttt)

        # mask invalid move, normalize probability
        # env.action_space() return list of tuple (collapse_block_id, free_qblock_ids)
        # which will be converted to action code
        valid_action_codes = encode_action_space(env.action_space())
        valid_policy_given_state = propagate_valid_action_probability_vector(
            valid_action_codes, policy_given_state)
        training_examples.append([state, valid_policy_given_state, None])

        # choose action
        action_code = np.random.choice(len(valid_policy_given_state), p=valid_policy_given_state)
        # decode_action_code covert an action code to (collapse_block_id, agent_move)
        # we use collapse_block_id to choose what is the collapsed qttt
        agent_move, collapsed_qttt = act(decode_action_code(action_code), env.action_space())

        # step action
        next_qttt, _, reward, done = env.step(agent_move, collapsed_qttt)

        if done:
            update_reward(training_examples, reward)
            break

        return training_examples


def propagate_valid_action_probability_vector(valid_action_codes, policy_given_state):
    pass


def encode_valid_move(agent_move, collapsed_qttt):
    pass


def act(action_code, agent_move, collapsed_qttt):
    pass


def update_reward(training_examples, reward):
    pass


def train(competitor_net, training_example):
    pass


def battle(env, competitor_net, curr_net, config):
    """
        compete_rounds: rounds of games for old/new network competition
        max_round_of_simulation: MCTS related parameters, # of rounds of planning before yielding
                    the final action
    """
    # we can use only 1 tree here
    competitor_mc = MTCS(competitor_net, config.exploration_level)
    curr_mc = MTCS(curr_net, config.exploration_level)
    monte_carlo_trees = [competitor_mc, curr_mc]

    score_board = {competitor_mc: 0, curr_mc: 0}

    for _ in range(config.compete_rounds):
        while True:
            curr_qttt, whose_turn = env.get_state_from_constant_view()

            mc = monte_carlo_trees[whose_turn]

            for _ in range(config.rounds_of_planning):
                mc.search(curr_qttt)

            state, policy_given_state = mc.get_policy_for_state(curr_qttt)

            # mask invalid move, normalize probability
            # action space [(collapse_choice, free_block_ids, collapsed_qttts)]
            valid_policy_given_state = propagate_valid_action_probability_vector(
                env.action_space(), policy_given_state)

            # choose action
            action_code = np.random.choice(len(valid_policy_given_state), p=valid_policy_given_state)
            agent_move, collapsed_qttt = act(action_code, env.action_space())

            # step action
            next_qttt, _, reward, done = env.step(agent_move, collapsed_qttt)

            if done:
                if reward > 0:
                    score_board[mc] += 1
                elif reward < 0:
                    score_board[curr_mc if mc == competitor_mc else competitor_mc] += 1
                # swap player sequence to make a different play take the first hand in the next round
                monte_carlo_trees.reverse()
                break

        return score_board[competitor_mc], score_board[curr_mc]


def save(competitor_net, *args):
    pass
