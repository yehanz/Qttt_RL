import numpy as np
from copy import deepcopy

REWARD = {
    'NO_REWARD': 0.0,
    'O_WIN_REWARD': 1.0,
    'X_WIN_REWARD': -1.0,
    # both O and X wins, but O wins earlier
    'OX_WIN_REWARD': 0.7,
    # both O and X wins, but X wins earlier
    'XO_WIN_REWARD': -0.7,
    'TIE_REWARD': 0.5,
}


class Env:
    def __init__(self):
        self.qttt = Qttt()
        self.round_ctr = 1

        self.collapsed_qttts = [Qttt()]
        self.next_valid_moves = [[i for i in range(9)]]

    def reset(self):
        self.qttt = Qttt()
        self.round_ctr = 1

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

    def step(self, qttt, agent_move, mark):
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
        self.round_ctr += 1

        self.qttt = qttt

        self.qttt.step(agent_move, mark)

        if self.qttt.has_cycle(agent_move, mark):
            self.collapsed_qttts = self.qttt.get_all_possible_collapse(agent_move, mark)
        else:
            self.collapsed_qttts = [self.qttt.copy()]

        self.next_valid_moves = []
        for qttt in self.collapsed_qttts:
            self.next_valid_moves.append(None if qttt.has_won()[0] else qttt.get_free_QBlock_ids())

        done, winner = self.qttt.has_won()

        reward = REWARD['NO_REWARD']
        if done and winner:
            # update reward here
            pass

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
        return self.next_valid_moves, self.collapsed_qttts

    def has_won(self):
        return self.qttt.has_won()


class Qttt:
    def __init__(self):
        self.board = [Qttt.QBlock(i) for i in range(9)]
        self.ttt = self.ttt()

    '''
    @classmethod
    def qttt_empty(cls) -> 'Qttt':
        return cls(board_state=[Qttt.QBlock(i) for i in range(9)], ttt=Qttt.ttt())

    @classmethod
    def qttt_with_state(cls, board_state):
        qttt = cls(board_state=board_state, ttt=Qttt.ttt())
        qttt.propagate_qttt_to_ttt()
        return qttt

    @classmethod
    def qttt_with_ttt_state(cls, board_state, ttt):
        return cls(board_state=board_state, ttt=ttt)
    '''

    def get_state(self):
        """
        State should be read only!
        :return:
        """
        return self.board

    def has_cycle(self):

        def get_graph_info(board):
            node_num = 0
            edges = []
            for block in board:
                if not block.mark and block.entangled_blocks:
                    node_num += 1
                    for node1 in block.entangled_blocks:
                        node2 = block.block_id
                        if node1 > node2:
                            continue
                        edges.append((node1, node2))
            return node_num, edges

        def validTree(n, edges):
            if n != len(edges) + 1:
                return False
            parent = list(range(n))

            def union(x, y):
                root_x = find(x)
                root_y = find(y)
                if root_x != root_y:
                    parent[root_y] = parent[root_x]

            def find(x):
                if x != parent[x]:
                    parent[x] = find(parent[x])
                return parent[x]

            for x, y in edges:
                union(x, y)
            return len({find(i) for i in range(n)}) == 1

        node_num, edges = get_graph_info(self.board)
        # print([node_num, edges])
        return not validTree(node_num, edges)

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
            if collapsed_block.mark != None:
                return
            entangled_block_ids, entangled_marks = collapsed_block.collapse(last_mark)
            for i in range(len(entangled_block_ids)):
                consequent_collapse(board, entangled_block_ids[i], entangled_marks[i])

        choice1, choice2 = last_move
        possible_collapse1 = deepcopy(self)
        consequent_collapse(possible_collapse1.board, choice1, last_mark)

        possible_collapse2 = deepcopy(self)
        consequent_collapse(possible_collapse2.board, choice2, last_mark)
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
        # put mark in pair locations
        loc1, loc2 = loc_pair
        self.board[loc1].place_mark(mark, loc2)
        self.board[loc2].place_mark(mark, loc1)
        # after step, always update corresponding ttt state
        # self.propagate_qttt_to_ttt()

    def visualize_board(self):
        # visualize the Qttt board
        for i in range(3):
            print("{:9s}|{:9s}|{:9s}".format(*[" ".join([str(integer) for integer in self.board[k].entangled_marks]) for k in range(i*3, i*3 + 3)]))

    '''
    def propagate_qttt_to_ttt(self):
        """
        Update ttt state with Qttt state, where a block of ttt is occupied by some mark
        only if the corresponding QBlock has collapsed

        :return:
            None
        """
        self.ttt.update_ttt_from_qttt(self.board)
    '''

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
            self.mark = None

        def place_mark(self, mark, entangle_block_id):
            self.entangled_marks.append(mark)
            self.entangled_blocks.append(entangle_block_id)

        def collapse(self, mark):
            """
            Collapse current block.

            :param mark: mark to be the collapsed mark in current block
            :return: list of entangled block number to collapse and marks
            """
            blocks_to_collapse = self.entangled_blocks[:]
            marks_to_collapse = self.entangled_marks[:]
            # remove current mark
            current_mark_pos = marks_to_collapse.index(mark)
            blocks_to_collapse.pop(current_mark_pos)
            marks_to_collapse.pop(current_mark_pos)
            self.entangled_blocks = []
            self.entangled_marks = []
            self.mark = mark
            return blocks_to_collapse, marks_to_collapse

    class ttt:
        def __init__(self):
            self.board = np.zeros(9, dtype=int)

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

        def update_ttt_from_qttt(self, Qttt_board):
            self.board, _ = Qttt.ttt.get_ttt_from_qttt(Qttt_board)

        @staticmethod
        def get_ttt_from_qttt(Qttt_board):
            """
            Update the state of ttt with Qttt.board.
            A collapsed QBlock can be characterized as
            - QBlock.entangled_blocks is empty
            - QBlock.marks contains only 1 element, which is the mark after collapse

            :param Qttt_board: state of Qttt
            :return:
                np.array: ttt.board
                list(int) free_block_ids: ids of free ttt block, where we can place
                spooky marks on the corresponding QBlocks.
            """
            pass

        def has_won(self):
            """
            Check if the game enters a terminal state.
            TODO: I think when a Qttt reached a terminal state, all of its block should have
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

            done = False
            winner = (5, 6)
            return done, winner

        def get_free_block_ids(self):
            return np.where(self.board == 0)[0]
