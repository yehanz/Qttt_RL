import math
import torch
from torch.distributions.dirichlet import Dirichlet

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
        # Add Dirichlet Noise to the new root node
        s = self.env.qttt.get_state()
        noise_dim = len(self.Vs[s])
        self.Ps[s][self.Vs[s]] += 0.25 * Dirichlet(torch.tensor([0.03] * noise_dim)).sample_n(noise_dim)

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

        self.env.change_to_even_pieces_view()
        s = self.env.qttt.get_state()

        counts = np.zeros(74)
        for action_code in self.env.get_valid_action_codes():
            counts[action_code] = self.Nsa[(s, action_code)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.array([0] * len(counts))
            probs[bestA] = 1
            return self.env.collapsed_qttts, probs

        # init situation
        counts = counts ** (1. / temp)
        counts_sum = counts.sum()
        probs = counts / counts_sum

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
            return -self.Es[s][1]

        if s not in self.Ps:
            # expand a new leaf node
            self.Ps[s], v = self.nn.predict(game_env)

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

        # s not leaf node
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in valids:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * \
                    math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        game_env.act(a)
        game_env.change_to_even_pieces_view()
        v = self.search(game_env)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
