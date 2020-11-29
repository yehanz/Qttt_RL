import math

from AlphaZero_Qttt.env_bridge import *

EPS = 1e-8

REWARD = {
    'NO_REWARD': 0.0,
    'Y_WIN_REWARD': 1.0,
    'X_WIN_REWARD': -1.0,
    # both Y and X wins, but Y wins earlier
    'YX_WIN_REWARD': 0.7,
    # both Y and X wins, but X wins earlier
    'XY_WIN_REWARD': -0.7,
    'TIE_REWARD': 0.0,
}


class MCTS:
    def __init__(self, env, nn, sim_nums, cpuct):
        self.env = env
        self.env.change_to_even_pieces_view()
        self.nn = nn
        self.sim_nums = sim_nums
        self.cpuct = cpuct

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores qttt.has_won() ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def step(self, action_code):
        # always append act with change to even piece view
        self.env.act(action_code)
        self.env.change_to_even_pieces_view()

    def get_action_prob(self, temp=1):
        """
        This function performs sim_nums simulations of MCTS starting from
        qttt.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.sim_nums):
            self.search(deepcopy(self.env))
            print('\n\n\n')

        s = self.env.qttt.get_state()

        ''' what is in the next_valid_moves, we need to choose valid actions here'''
        counts = np.zeros(74)
        for action_code in self.env.get_valid_action_codes():
            counts[action_code] = self.Nsa[(s, action_code)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        # init situation
        counts = counts ** (1. / temp)
        counts_sum = counts.sum()
        assert counts_sum != 0
        probs = counts / counts_sum

        self.env.change_to_even_pieces_view()
        # probs not 74 length!
        return self.env.collapsed_qttts, probs

    def search(self, game_env: EnvForDeepRL):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        # there is only even piece view, at all time
        game_env.change_to_even_pieces_view()
        s = game_env.qttt.get_state()
        done, winner = game_env.has_won()
        reward = REWARD[winner + '_REWARD']

        if s not in self.Es:
            self.Es[s] = (done, reward)
        if self.Es[s][0]:
            print('4444444444 BP END STATE 444444444')
            print('round counter and player id')
            print(game_env.round_ctr, game_env.player_id)
            print('state and value')
            print(s, self.Es[s][1])
            # terminal node
            return -self.Es[s][1]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nn.predict(game_env)

            '''
            Need to check whether the probability vector returned by self.nn.predict() is masked or not

            '''
            valids = game_env.get_valid_action_codes()

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, doing a workaround.")

            self.Vs[s] = valids  # store valid moves given state s
            self.Ns[s] = 0
            print('2222222 expand and BP leaf node 222222222')
            print('round counter and player id')
            print(game_env.round_ctr, game_env.player_id)
            print('state and value')
            print(s, v)
            return -v

        # not leaf node
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in valids:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * \
                    math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = EPS + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s])  # Q = 0 ?, supposed to be an EPS

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        print('1111111111 select 11111111111111')
        print('before: state, round ctr, player id, action')
        print(s)
        print(game_env.round_ctr, game_env.player_id)
        if a > 71:
            print('select a collapsed ending state! %d' % a)
        print(INDEX_TO_MOVE[a % 36], a)
        game_env.act(a)
        print('after: state, round ctr, player id')
        print(game_env.qttt.get_state())
        print(game_env.round_ctr, game_env.player_id)

        game_env.change_to_even_pieces_view()
        print('even piece view')
        print(game_env.qttt.get_state())
        print(game_env.round_ctr, game_env.player_id)
        # TODO: How to handle case where one of the collapse case is the terminate state?
        # In that case, one of the next valid move list is None
        v = self.search(game_env)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        # print('444444444 back propagate PARENT 44444444')
        # print('state and value')
        # print(s, v)
        return -v
