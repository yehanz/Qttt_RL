from env import Env
import numpy as np
from copy import deepcopy

INDEX_TO_MOVE = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 5), (4, 6), (4, 7), (4, 8),
    (5, 6), (5, 7), (5, 8),
    (6, 7), (6, 8),
    (7, 8),
]

MOVE_TO_INDEX = [
    [-1, 0, 1, 2, 3, 4, 5, 6, 7],
    [-1, 8, 9, 10, 11, 12, 13, 14],
    [-1, 15, 16, 17, 18, 19, 20],
    [-1, 21, 22, 23, 24, 25],
    [-1, 26, 27, 28, 29],
    [-1, 30, 31, 32],
    [-1, 33, 34],
    [-1, 35],
    [-1]
]


class EnvForDeepRL(Env):

    def __init__(self):
        super(EnvForDeepRL, self).__init__()
        # if this env is a constant view converted from other game env
        # then this attribute indicates the bias added to each piece
        # on the chess board of the current env
        self.player_id = 1

    @property
    def current_player_id(self):
        return self.player_id

    @property
    def valid_action_mask(self):
        def valid_moves_to_mask(free_block_ids):
            mask = np.zeros(36)
            lenght = len(free_block_ids)
            for i in range(lenght):
                for j in range(i, lenght):
                    mask[MOVE_TO_INDEX[i][j]] = 1

        mask = np.zeros([])
        for free_blocks in self.next_valid_moves:
            mask = np.concatenate((mask, valid_moves_to_mask(free_blocks)))

        if len(self.next_valid_moves) == 1:
            mask = np.concatenate((mask, mask))
        return mask

    def _flip_odd_and_even(self):
        self.round_ctr ^= 1
        self.qttt.flip_odd_and_even(1)
        for qttt in self.collapsed_qttts:
            qttt.flip_odd_and_even(1)

    def change_to_even_pieces_view(self):
        if self.player_id != 0:
            self._flip_odd_and_even()

    def change_to_normal_view(self):
        # in normal view, round_ctr and player_id share the same oddity
        if (self.round_ctr & 1) ^ self.player_id == 1:
            self._flip_odd_and_even()

    def index_to_agent_move(self, prob_vector_index):
        collapsed_qttt_idx = prob_vector_index % 36
        index_0_to_35 = collapsed_qttt_idx - 36 * (collapsed_qttt_idx % 36)
        return self.collapsed_qttts[collapsed_qttt_idx], INDEX_TO_MOVE[index_0_to_35]

    def pick_a_valid_move(self, net_prob_vector):
        # net_prob_vector = net_prob_vector * self.valid_action_mask
        # valid_prob_vector = net_prob_vector / net_prob_vector.sum()

        # choose action
        action_code = np.random.choice(len(net_prob_vector), p=net_prob_vector)
        return action_code

    def act(self, action_code):
        # decode_action_code covert an action code to (collapse_block_id, agent_move)
        # we use collapse_block_id to choose what is the collapsed qttt
        collapsed_qttt, agent_move = self.index_to_agent_move(action_code)
        self.player_id = self.player_id ^ 1
        return super().step(collapsed_qttt, agent_move)

