from collections import deque
from copy import deepcopy

import numpy as np
import torch

# X is the first player
REWARD = {
    'NO_REWARD': 0.0,
    'EVEN_WIN_REWARD': 1.0,
    'ODD_WIN_REWARD': -1.0,
    # both Y and X wins, but Y wins earlier
    'EVEN_ODD_WIN_REWARD': 0.7,
    # both Y and X wins, but X wins earlier
    'ODD_EVEN_WIN_REWARD': -0.7,
    'TIE_REWARD': 0.0,
}


class Env:
    def __init__(self):
        self.qttt = Qttt()
        self.round_ctr = 1
        # player id is either 0 or 1
        self.player_id = 0

        self.collapsed_qttts = [Qttt()]
        self.next_valid_moves = [np.arange(9)]
        self.collapse_choice = ()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.qttt != other.qttt or self.collapsed_qttts != other.collapsed_qttts or \
                    self.collapse_choice != other.collapse_choice:
                return False
            return True
        return False

    @property
    def current_player_id(self):
        return self.player_id

    def reset(self):
        self.qttt = Qttt()
        self.round_ctr = 1
        self.player_id = 0

        self.collapsed_qttts = [Qttt()]
        self.next_valid_moves = [np.arange(9)]
        self.collapse_choice = ()
        return self.qttt, self.round_ctr

    def get_state(self):
        """
        Get current state of Qttt board
        :return:
            Qttt: qttt object of current Env
            int mark: we use env.round_ctr as our mark for current round of play
        ATTENTION: we return reference instead of a copy of current state here, it
                is supposed to be READ ONLY. The state of Env should only be modified
                by env.step().
        """

        return self.qttt, self.round_ctr

    def step(self, qttt, agent_move, mark=None):
        """
        Carry out actions taken by agents
        :param Qttt             collapsed_qttt: qttt object with collapsed state
                                                an agent chooses
        :param tuple(int, int)  agent_move: pair of qblock ids to place spooky marks
        :param int              mark: mark used for action this round
        :return:
            Qttt    next_qttt: updated qttt object after step
            int     next_mark: mark to use for the next round
            int     reward: immediate reward for current action
            boolean done: if the game has reached the terminal state
        """
        if not mark:
            mark = self.round_ctr

        # update states
        self.round_ctr += 1
        self.player_id = self.player_id ^ 1

        self.qttt = qttt
        self.qttt.step(agent_move, mark)

        if self.qttt.has_cycle:
            self.collapsed_qttts = self.qttt.get_all_possible_collapse(agent_move, mark)
            self.collapse_choice = agent_move
        else:
            self.collapsed_qttts = [deepcopy(self.qttt)]
            self.collapse_choice = ()

        self.next_valid_moves = []
        for qttt in self.collapsed_qttts:
            self.next_valid_moves.append(None if qttt.has_won()[0] else qttt.get_free_QBlock_ids())

        done, winner = self.qttt.has_won()
        reward = REWARD[winner + '_REWARD']

        return self.qttt, self.round_ctr, reward, done

    def get_valid_moves(self):
        """
        Get all valid moves based on current action
        :return:
            list(list(int)) next_valid_moves: each sub list contains ids of
                free QBlock under a given collapsed_qttt, where next_valid_moves[i]
                corresponds to id of free QBlocks for collapsed_qttt[i]
            list(Qttt)      collapsed_qttt: possible states of Qttt.board after collapse
        """
        return self.next_valid_moves, self.collapsed_qttts, self.collapse_choice

    def has_won(self):
        return self.qttt.has_won()

    def render(self):
        self.qttt.visualize_board()


class Qttt:

    def __init__(self):
        self.board = [Qttt.QBlock(i) for i in range(9)]
        self.ttt = Qttt.ttt()
        self.has_cycle = False

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            for q1, q2 in zip(self.board, other.board):
                if (q1.entangled_marks != q2.entangled_marks) or \
                        (q1.entangled_blocks != q2.entangled_blocks):
                    return False
            return True
        return False

    def add_bias_to_pieces(self, bias):
        for qblock in self.board:
            qblock.add_bias_to_pieces(bias)
        self.ttt.add_bias_to_pieces(bias)

    def get_state(self):
        """
        State should be read only!
        :return:
        """
        return self.to_hashable()

    def _has_cycle(self, agent_move, mark):
        # bfs to find cycle
        start_point_id, end_point_id = agent_move
        visited = set([start_point_id])
        q = deque([start_point_id])
        while q:
            cur_point_id = q.popleft()
            cur_point = self.board[cur_point_id]
            for i in range(len(cur_point.entangled_blocks)):
                entangled_block = cur_point.entangled_blocks[i]
                entangled_mark = cur_point.entangled_marks[i]
                if entangled_block == end_point_id:
                    if entangled_mark != mark:
                        return True
                    else:
                        continue
                if entangled_block in visited:
                    continue
                else:
                    visited.add(entangled_block)
                    q.append(entangled_block)
        return False

    def get_all_possible_collapse(self, last_move, last_mark):
        """
        Get all possible collapse based on current Qttt chess board state
        Only used if Qttt.has_cycle returns True.
        There should only be 2 possible collapse, a simple way to understand this is as follows:
            We constantly place marks on the Qttt board, until some round, we place a spooky marks M
            over a pair of QBlocks(b1,b2) and a entangled cycle forms, there is 2 way to collapse this
            cycle: M collapse at QBlock b1, or M collapse at QBlock b2
        Idea for implementation: BFS
            take a look at special example
            3   2       1,9
            4   1,2,3   8
                4,5,6
                7,8,9
            5   6       7
            We can see that QBlock 5 and 3 forms an entangled cycle, in this case
            last_move = (3,5), last_mark = 9
            We choose 9 to collapse either at QB 3 or QB 5
            This is the start point of BFS.
            If we choose mark 9 collapse at QBlock 5. It is also noticed that QBlock 5 has many other spooky
            marks, once this QBlock is collapsed, the rest of the spooky marks in the same block collapses
            automatically(like an avalanche, that's BFS, from the start point of the collapse we cause many
            other blocks to collapse as well.
            Finally, we get 2 possible collapsed Qttt state:
            3   2   9       3   2   1
            4   1   8   or  4   9   8
            5   6   7       5   6   7
        :param tuple(int, int) last_move: last move that forms the entangle cycle
        :param int last_mark: last mark used for last move
        :return:
            list(Qttt): list of Qttt objects after collapse.
        """

        def consequent_collapse(board, collapsed_block_id, last_mark):
            collapsed_block = board[collapsed_block_id]
            if not collapsed_block.entangled_blocks and collapsed_block.entangled_marks:
                return
            entangled_block_ids, entangled_marks = collapsed_block.collapse(last_mark)
            for i in range(len(entangled_block_ids)):
                consequent_collapse(board, entangled_block_ids[i], entangled_marks[i])

        choice1, choice2 = last_move

        possible_collapse1 = deepcopy(self)
        consequent_collapse(possible_collapse1.board, choice1, last_mark)
        # always update corresponding ttt state and has cycle flag
        possible_collapse1.propagate_qttt_to_ttt()
        possible_collapse1.has_cycle = False

        possible_collapse2 = deepcopy(self)
        consequent_collapse(possible_collapse2.board, choice2, last_mark)
        # always update corresponding ttt state and has cycle flag
        possible_collapse2.propagate_qttt_to_ttt()
        possible_collapse2.has_cycle = False

        possible_collapse = [possible_collapse1, possible_collapse2]
        return possible_collapse

    def step(self, loc_pair, mark):
        """
        Step func modify the Qttt chess board state
        :param Qttt collapsed_qttt: it specifies what collapsed Qttt
                state that current action is based on
        :param tuple(int, int) loc_pair: each element in tuple represents the QBlock id
                where the spooky mark is placed. It can be None if collapsed_qttt already
                reach the terminal state of the game
        :param int mark: mark used for this round of play. It is essentially env.ctr
        :return:
            self
        """
        if loc_pair is None:
            self.has_cycle = False
            return
        # put mark in pair locations
        loc1, loc2 = loc_pair

        self.board[loc1].place_mark(mark, loc2)
        self.board[loc2].place_mark(mark, loc1)
        self.has_cycle = self._has_cycle(loc_pair, mark)

    def visualize_board(self):
        # visualize the Qttt board
        for i in range(9):
            if not self.board[i].has_collapsed:
                print("{:17s}|".format(" ".join([str(integer) for integer in self.board[i].entangled_marks])), end="")
            else:
                print("{:16s}*|".format(str(self.board[i].collapsed_mark)), end="")
            if i % 3 == 2:
                print("")
        print("")

    # !FIX: come up with a faster way: bytes([1,2,3]) might be faster
    def to_hashable(self):
        board = []
        for i in range(9):
            if not self.board[i].has_collapsed:
                block = tuple(self.board[i].entangled_marks)
                board.append(block)
            else:
                block = (self.board[i].collapsed_mark, 0)
                board.append(block)
        return tuple(board)

    def propagate_qttt_to_ttt(self):
        """
        Update ttt state with Qttt state, where a block of ttt is occupied by some mark
        only if the corresponding QBlock has collapsed
        :return:
            None
        """
        # update ttt board
        for i in range(9):
            self.ttt.board[i] = self.board[i].collapsed_mark \
                if self.board[i].has_collapsed else self.ttt.board[i]

    def get_free_QBlock_ids(self):
        """
        Find ids for QBlocks which haven't collapsed
        :return:
            np.array(int): array of QBlock ids
        """
        return self.ttt.get_free_block_ids()

    def has_won(self):
        # self.propagate_qttt_to_ttt()
        return self.ttt.has_won()

    def show_result(self):
        done, winner = self.has_won()
        if not done:
            print("Game does not end")
            return
        print("Game ends.")
        if winner == "ODD_WIN":
            print("ODD wins!")
        elif winner == "EVEN_WIN":
            print("EVEN wins!")
        elif winner == "ODD_EVEN_WIN":
            print("Both ODD and EVEN win, but ODD wins earlier!")
        elif winner == "EVEN_ODD_WIN":
            print("Both ODD and EVEN win, but EVEN wins earlier!")
        elif winner == "TIE":
            print("Tie!")
        else:
            print(winner)

    def to_tensor(self):
        """
        convert qttt to a 10*3*3 tensor
        first channel represents collapsed block state, 0 means block hasn't collapsed
            non-zero block represents piece that collapses at this location

        last 10 channels:
            for collapsed block, all channels are 0
            otherwise for piece with num k presented at Qblock (i,j), then tensor(k,i,j)=1
            else 0
        :return:
        """
        tensor_repr = torch.zeros(9, 11)
        for idx, qblock in enumerate(self.board):
            tensor_repr[idx][0] = qblock.has_collapsed
            if qblock.entangled_marks:
                tensor_repr[idx][qblock.entangled_marks] = 1
        return tensor_repr.t().reshape(11, 3, 3)

    class QBlock:
        def __init__(self, block_id):
            """
            Qttt chess board state representation
            If X put pieces at 3rd and 5th block, then for 3rd Qblock, add
            entangled block number and symbols that X uses during this round, which
            is just env.round_ctr
            entangled_blocks list(int): blocks that share the same spooky marks put in current block
            marks: marks that current block contains
            we can get the number of the other block for any spooky marks self.marks[i] put in current block
            with self.entangled_blocks[i]
            we use a compact representation to distinguish between spooky marks and collapsed marks:
            spooky marks: if self.marks[i] is a spooky mark, self.entangled_blocks[i] != self.block_id
            collapsed marks: if self.marks[i] is a collapsed mark, then we have
                    - i = 0
                    - len(self.marks) = 1
                    - len(self.entangled_blocks) = 0
                    - at the same block_id in ttt() we have the same mark
            """
            self.entangled_blocks = []
            self.entangled_marks = []
            self.block_id = block_id

        @property
        def has_collapsed(self):
            return bool(self.entangled_marks and not self.entangled_blocks)

        @property
        def collapsed_mark(self):
            return self.entangled_marks[0] if self.has_collapsed else None

        def add_bias_to_pieces(self, bias):
            self.entangled_marks = [mark + bias for mark in self.entangled_marks]

        def place_mark(self, mark, entangle_block_id):
            self.entangled_marks.append(mark)
            self.entangled_blocks.append(entangle_block_id)

        def collapse(self, mark):
            """
            Collapse current block.
            :param mark: mark to be the collapsed mark in current block
            :return: list of entangled block number to collapse and marks
            """
            blocks_to_collapse = self.entangled_blocks
            marks_to_collapse = self.entangled_marks
            # remove current mark
            current_mark_pos = marks_to_collapse.index(mark)
            blocks_to_collapse.pop(current_mark_pos)
            marks_to_collapse.pop(current_mark_pos)
            self.entangled_blocks = []
            self.entangled_marks = [mark]
            return blocks_to_collapse, marks_to_collapse

    class ttt:
        EMPTY_BLOCK_VAL = 0

        def __init__(self):
            self.board = np.ones(9, dtype=int) * Qttt.ttt.EMPTY_BLOCK_VAL

        '''
        def step(self, loc, mark):
            """
            place a piece at given location
            :param int  loc: 1~9 location id of the block
                                1 2 3
                                4 5 6
                                7 8 9
            :param int  mark: use env.round_ctr as mark. Odd num for 'O', Even num for 'X'
            :return:
                int reward: reward for current move
                int winner_mark: Odd value for 'O', Even value for 'X'
            """
            self.board[loc] = mark
            return self.has_won()
        '''

        def add_bias_to_pieces(self, bias):
            self.board = np.where(self.board == Qttt.ttt.EMPTY_BLOCK_VAL,
                                  self.board, self.board + bias)

        def has_won(self):
            """
            Check if the game enters a terminal state.
            When a Qttt reached a terminal state, all of its block should have
            collapsed, so we are able to determine if it is a tie, a winner or 2 winners from
            the ttt state only.
            :param ttt: current ttt state of Qttt
            :return:
                boolean done: if game has reached the terminal state
                list(int) winners: if tie, winners is None
                    if not a tie, check game rules, several examples below:
                    1 3 5                   1 3 7
                    2 4 6 => winner(5,6)    2 5 4  => winner(7) TODO: Or maybe both 7 and 9?
                    - - -                   6 8 9
            """

            def tictactoe(moves):
                """
                :type moves: List[List[int]]
                :rtype: str
                """
                rows = [0] * 3
                cols = [0] * 3
                diag = 0
                disdiag = 0
                max_x = 0
                max_y = 0
                occupied_block_num = 0
                for idx in range(len(moves)):
                    if moves[idx] == Qttt.ttt.EMPTY_BLOCK_VAL:
                        continue
                    occupied_block_num += 1
                    if moves[idx] % 2 == 1:
                        max_x = max(max_x, moves[idx])
                    else:
                        max_y = max(max_y, moves[idx])
                    i = idx // 3
                    j = idx % 3
                    rows[i] += (moves[idx] % 2) * 2 - 1
                    cols[j] += (moves[idx] % 2) * 2 - 1
                    if i + j == 2:
                        disdiag += (moves[idx] % 2) * 2 - 1
                    if i == j:
                        diag += (moves[idx] % 2) * 2 - 1
                odd_win = (3 in rows) or (3 in cols) or diag == 3 or disdiag == 3
                even_win = (-3 in rows) or (-3 in cols) or diag == -3 or disdiag == -3
                if odd_win and even_win:
                    if max_x < max_y:
                        winner = 'ODD_EVEN_WIN'
                    else:
                        winner = 'EVEN_ODD_WIN'
                elif odd_win:
                    winner = 'ODD_WIN'
                elif even_win:
                    winner = 'EVEN_WIN'
                else:
                    if occupied_block_num >= 8:
                        winner = "TIE"
                    else:
                        winner = "NO"
                return winner

            winner = tictactoe(self.board)
            done = True if winner != "NO" else False
            return done, winner

        def get_free_block_ids(self):
            return np.where(self.board == Qttt.ttt.EMPTY_BLOCK_VAL)[0]

        def visualize_board(self):
            # visualize the ttt board
            for i in range(3):
                print("{:2d}|{:2d}|{:2d}".format(*[self.board[k] for k in range(i * 3, i * 3 + 3)]))


def after_action_state(collapsed_qttt, action, mark):
    board = deepcopy(collapsed_qttt)
    if action is None:
        return board.to_hashable()
    board.step(action, mark)
    return board.to_hashable()
