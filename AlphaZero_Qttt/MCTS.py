import numpy as np
from copy import deepcopy
from AlphaZero_Qttt.Network import Network
from AlphaZero_Qttt.env_bridge import *
import math

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
    def __init__(self, env, nn, sim_nums, cpuct, tree_id):
        self.env = env
        self.nn = nn
        self.sim_nums = sim_nums
        self.cpuct = cpuct
        self.tree_id = tree_id

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores qttt.has_won() ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
    
    def step(self, action_code):
        self.env.act(action_code) 


    def get_action_prob(self, temp=1):
        """
        This function performs sim_nums simulations of MCTS starting from
        qttt.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.sim_nums):
            self.search(deepcopy(self.env.qttt))

        s = self.env.qttt.get_state()

        ''' what is in the next_valid_moves, we need to choose valid actions here'''
        
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in self.env.get_valid_action_codes()]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        #init situation
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]

        self.env.change_to_even_pieces_view()
        if len(self.env.collapsed_qttts) < 2:
            self.env.collapsed_qttts.append(self.env.collapsed_qttts.append[0])
        return self.env.collapsed_qttts, probs


    def search(self, qttt):
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

        s = qttt.get_state()
        done, winner = qttt.has_won()
        reward = REWARD[winner + '_REWARD']

        if s not in self.Es:
            self.Es[s] = (done, reward)
        if self.Es[s][0]:
            # terminal node
            return self.Es[s][1] if self.env.current_player_id == self.tree_id else -self.Es[s][1]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nn.predict(self.env)
            
            '''
            Need to check whether the probability vector returned by self.nn.predict() is masked or not

            '''
            valids = self.env.get_valid_action_codes()

            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                print("All valid moves were masked, doing a workaround.")


            self.Vs[s] = valids # store valid moves given state s
            self.Ns[s] = 0
            return v if self.env.current_player_id == self.tree_id else -v

        # not leaf node
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in valids:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?, supposed to be an EPS

            if u > cur_best:
                cur_best = u
                best_act = a

        action_code = best_act

        #Do we need to deepcopy self.env here?
        env_temp= deepcopy(self.env)
        env_temp.change_to_even_pieces_view() 
        next_qttt, _, _, _ = env_temp.act(action_code)
        
        v = self.search(next_qttt)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v if self.env.current_player_id == self.tree_id else -v


