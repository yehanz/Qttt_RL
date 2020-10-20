from collections import defaultdict


class GameTree:

    # note when accessing an unseen state, state count returns 1
    # so that you don't have to increase state counter by 1 when update the state value
    state_val = defaultdict(lambda x: 1.0)
    state_cnt = defaultdict(lambda x: 1)

    @staticmethod
    def set_state_value(state, val):
        """Set state value for a given state

        >>> val = GameTree.get_stat_val(state) + error / GameTree.get_state_cnt(state)
        >>> GameTree.set_state_value(state, val)

        ATTENTION! It is noted that no need to increase the state counter when calculate state value!
        Since each time we call set_state_val, counter is increased for next time use.

        Args:
            state(BoardState): state of the board
            val(float): state value

        Returns:
            None
        """
        # increase the state counter for next time use
        GameTree.state_cnt[state] += 1
        GameTree.state_val[state] = val

    @staticmethod
    def get_stat_val(state):
        return GameTree.state_val[state]

    @staticmethod
    def get_state_cnt(state):
        return GameTree.state_cnt[state]

    @staticmethod
    def reset_game_tree():
        GameTree.state_val = defaultdict(lambda x: 1)
        GameTree.state_cnt = defaultdict(lambda x: 1)
