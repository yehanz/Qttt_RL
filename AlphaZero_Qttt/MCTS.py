from collections import defaultdict

import math

from AlphaZero_Qttt.env_bridge import *
from env import REWARD

EPS = 1e-8


class MCTS:
    def __init__(self, env, nn, sim_nums, cpuct):
        self.env = env
        self.env.change_to_even_pieces_view()
        self.nn = nn
        self.sim_nums = sim_nums
        self.cpuct = cpuct

        self.Qsa = defaultdict(lambda: 0)  # stores Q values for s,a (as defined in the paper)
        self.Nsa = defaultdict(lambda: 0)  # stores #times edge s,a was visited
        self.Ns = defaultdict(lambda: 0)  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores qttt.has_won() ended for board s
        self.Vs = {}  # # store valid moves given state s

        # initialize blank chess board node
        self.search(deepcopy(env))
        self.env.change_to_even_pieces_view()
        s = self.env.qttt.get_state()
        self.Ps[s] = self.add_dirichlet_noise(self.Vs[s], self.Ps[s])

    def reset_game_env(self):
        self.env = EnvForDeepRL()
        self.env.change_to_even_pieces_view()

    def step(self, action_code, is_train=True):
        # always append act with change to even piece view
        self.env.act(action_code)
        self.env.change_to_even_pieces_view()
        # Add Dirichlet Noise to the new root node to ensure exploration is always guarenteed
        s = self.env.qttt.get_state()
        # During battle no noise is added in order to perform the strongest play
        if s in self.Vs and is_train:
            self.Ps[s] = self.add_dirichlet_noise(self.Vs[s], self.Ps[s])

    def add_dirichlet_noise(self, valid_action_idx, act_probs):
        valid_child_priors = act_probs[valid_action_idx]  # select only legal moves entries in act_probs array
        valid_child_priors = 0.75 * valid_child_priors + \
                             0.25 * np.random.dirichlet(
            np.array([0.03] * len(valid_child_priors), dtype=np.float32))
        act_probs[valid_action_idx] = valid_child_priors
        return act_probs

    def get_action_prob(self, temp=1):
        """
        This function performs self.sim_nums simulations of MCTS starting from
        qttt.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        # for _ in tqdm(range(self.sim_nums)):
        for _ in range(self.sim_nums):
            self.search(deepcopy(self.env))

        self.env.change_to_even_pieces_view()
        s = self.env.qttt.get_state()

        # get_valid_action_codes() will return the valid action code given the current environment
        counts = np.zeros(74)
        for action_code in self.env.get_valid_action_codes():
            counts[action_code] = self.Nsa[(s, action_code)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.array([0] * len(counts))
            probs[bestA] = 1.0
            return deepcopy(self.env.collapsed_qttts), probs

        counts = counts ** (1. / temp)
        # counts_sum is the total number that state s is visited
        counts_sum = counts.sum()
        probs = counts / counts_sum

        return deepcopy(self.env.collapsed_qttts), probs

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

        ############## EXPAND and BP ##################

        if s not in self.Es:
            self.Es[s] = (done, reward)
        if self.Es[s][0]:
            return -self.Es[s][1]

        if s not in self.Ps:
            # expand a new leaf node
            self.Ps[s], v = self.nn.predict(game_env)

            # the valid action_codes given the current environment
            valids = game_env.get_valid_action_codes()

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # All valid moves may be masked if either your NNet architecture is
                # insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay
                # attention to your NNet and/or training process.
                print("All valid moves were masked, doing a workaround.")

            self.Vs[s] = valids  # store valid moves given state s
            self.Ns[s] = 0
            return -v

        ################ SELECT ####################
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in valids:
            # if an action is never tried before, default dict will return 0
            # for Qsa, Nsa, Ns
            # EPS is used here such that u is not all 0 if none of the action code
            # of s has been tried before, thus u = k* Psa, and network evaluation
            # can give us hint on the initial try.
            u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * \
                math.sqrt(self.Ns[s] + EPS) / (1 + self.Nsa[(s, a)])

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        game_env.act(a)
        game_env.change_to_even_pieces_view()
        # keep select-select-select until expand and bp
        v = self.search(game_env)

        ############### BP FROM Child Node #################
        # Updating the Qsa, Nsa, and Ns
        self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
        self.Nsa[(s, a)] += 1
        # Ns is the total count of trying child nodes
        self.Ns[s] += 1
        return -v
