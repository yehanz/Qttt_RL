import numpy as np

from env import Env

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
    [-1, -1, 8, 9, 10, 11, 12, 13, 14],
    [-1, -1, -1, 15, 16, 17, 18, 19, 20],
    [-1, -1, -1, -1, 21, 22, 23, 24, 25],
    [-1, -1, -1, -1, -1, 26, 27, 28, 29],
    [-1, -1, -1, -1, -1, -1, 30, 31, 32],
    [-1, -1, -1, -1, -1, -1, -1, 33, 34],
    [-1, -1, -1, -1, -1, -1, -1, -1, 35],
]


class EnvForDeepRL(Env):

    def __init__(self):
        super(EnvForDeepRL, self).__init__()
        self._dulplicate_collapsed_qttts_and_valid_moves()

    def _dulplicate_collapsed_qttts_and_valid_moves(self):
        self.collapsed_qttts.append(self.collapsed_qttts[0])
        self.next_valid_moves.append(self.next_valid_moves[0])

    @property
    def valid_action_mask(self):
        mask = np.zeros([72])
        mask[self.get_valid_action_codes()] = 1
        return mask

    def get_valid_action_codes(self):
        valid_act_codes = np.array([], dtype=int)
        for idx, free_block_ids in enumerate(self.next_valid_moves):
            lenght = len(free_block_ids)
            valid_act_codes = np.concatenate(
                (valid_act_codes,
                 np.array([MOVE_TO_INDEX[free_block_ids[i]][free_block_ids[j]]
                           for i in range(lenght) for j in range(i + 1, lenght)]) + 36 * idx)
            )
        return valid_act_codes

    def _add_bias_to_pieces(self, bias):
        self.round_ctr += bias
        self.qttt.add_bias_to_pieces(bias)
        for qttt in self.collapsed_qttts:
            qttt.add_bias_to_pieces(bias)

    def change_to_even_pieces_view(self):
        # normal view, odd piece's turn
        if self.round_ctr & 1 != 0:
            if self.player_id != 0:
                self._add_bias_to_pieces(1)
            elif self.player_id == 0:
                self._add_bias_to_pieces(-1)

    def change_to_normal_view(self):
        # in normal view, round_ctr and player_id share the same oddity
        if (self.round_ctr & 1) ^ self.player_id == 1:
            self._add_bias_to_pieces(-1)

    def index_to_agent_move(self, prob_vector_index):
        collapsed_qttt_idx = prob_vector_index // 36
        index_0_to_35 = prob_vector_index % 36
        return self.collapsed_qttts[collapsed_qttt_idx], INDEX_TO_MOVE[index_0_to_35]

    def pick_a_valid_move(self, net_prob_vector):
        # choose action
        action_code = np.random.choice(len(net_prob_vector), p=net_prob_vector)
        return action_code

    def act(self, action_code):
        # decode_action_code covert an action code to (collapse_block_id, agent_move)
        # we use collapse_block_id to choose what is the collapsed qttt
        collapsed_qttt, agent_move = self.index_to_agent_move(action_code)
        self.player_id = self.player_id ^ 1
        qttt, round_ctr, reward, done = super().step(collapsed_qttt, agent_move)
        if not qttt.has_cycle:
            self._dulplicate_collapsed_qttts_and_valid_moves()
        return qttt, round_ctr, reward, done
