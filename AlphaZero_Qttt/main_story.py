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
        max_round_of_planning: MCTS related parameters, # of rounds of planning before yielding
                    the final action
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
        new_wins, old_wins = battle(competitor_net, curr_net, config)
        if new_wins / (new_wins + old_wins) > config.min_win_rate_network_update:
            # save current network
            save(competitor_net)
            # use new network to generate training data
            curr_net = competitor_net
