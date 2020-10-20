import numpy as np

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

        self.collapsed_states = [self.qttt.board.copy()]
        self.next_valid_moves = [[i for i in range(9)]]

    def reset(self):
        self.qttt = Qttt()
        self.round_ctr = 1

    def get_state(self):
        """
        Get current state of Qttt board
        :return:
            Qttt.board: current board state, which is just a list of QBlock
            int mark: we use env.round_ctr as our mark for current round of play

        ATTENTION: we return reference instead of a copy of current state here, it
                is supposed to be READ ONLY. The state of Env should only be modified
                by env.step().
        """

        return self.qttt.board, self.round_ctr

    def step(self, collapsed_qttt_state, agent_move, mark):
        """
        Carry out actions taken by agents

        :param Qttt.board       collapsed_qttt_state: collapsed state an agent chooses
        :param tuple(int, int)  agent_move: pair of qblock ids to place spooky marks
        :param int              mark: mark used for action this round
        :return:
            Qttt.board  next_state: updated state for Qttt board
            int         next_mark: mark to use for the next round
            int         reward: immediate reward for current action
            boolean     done: if the game has reached the terminal state
        """
        self.round_ctr += 1

        next_state = self.qttt.step(collapsed_qttt_state, agent_move, mark)

        if self.qttt.has_cycle(agent_move, mark):
            self.collapsed_states = self.qttt.get_all_possible_collapse(agent_move, mark)
        else:
            self.collapsed_states = [self.qttt.board.copy()]
        self.next_valid_moves = [Qttt.ttt.get_ttt_from_qttt(states) for states in self.collapsed_states]

        self.qttt.propagate_qttt_to_ttt()
        done, winner = self.qttt.has_won()

        reward = REWARD['NO_REWARD']
        if done and winner:
            # update reward here
            pass

        return next_state, self.round_ctr, reward, done

    def get_valid_moves(self):
        """
        Get all valid moves based on current action

        :return:
            list(list(int)) next_valid_moves: each sub list contains ids of
                free QBlock under a given collapsed_states, where next_valid_moves[i]
                corresponds to id of free QBlocks for collapsed_states[i]
            Qttt.board      collapsed_states: possible states of Qttt.board after collapse
        """
        return self.next_valid_moves, self.collapsed_states

    def has_won(self):
        return self.qttt.has_won()


class Qttt:
    def __init__(self):
        self.board = [self.QBlock(i) for i in range(9)]
        self.ttt = Qttt.ttt()

    def has_cycle(self, last_move, last_mark):
        """
        Get cycle entanglement

        :param tuple(int, int) last_move: last action taken by the agent, for each element
                inside the tuple, it represents the block id of QBlock
        :return:
            True if there is one
            False otherwise
        """
        pass

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
            list(Qttt.board): list of Qttt.board after collapse.
        """

        return []

    def step(self, collapsed_choice, loc_pair, mark):
        """
        Step func modify the Qttt chess board state

        :param Qttt.board collapsed_choice: if not None, it specifies what collapsed Qttt
                state that current action is based on
        :param tuple(int, int) loc_pair: each element in tuple represents the QBlock id
                where the spooky mark is placed
        :param int mark: mark used for this round of play. It is essentially env.ctr
        :return:
        """
        pass

    def propagate_qttt_to_ttt(self):
        """
        Update ttt state with Qttt state, where a block of ttt is occupied by some mark
        only if the corresponding QBlock has collapsed

        :return:
            None
        """
        self.ttt.update_ttt_from_qttt(self.board)

    def has_won(self):
        self.propagate_qttt_to_ttt()
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
            self.marks = []
            self.block_id = block_id

        def place_mark(self, mark, entangle_block_id):
            pass

        def collapse(self, mark):
            """
            Collapse current block.

            :param mark: mark to be the collapsed mark in current block
            :return: list of entangled block number to collapse as well
            """
            blocks_to_collapse = self.entangled_blocks
            self.entangled_blocks = []
            assert self.marks.count(mark) > 0
            self.marks = [mark]

            return blocks_to_collapse

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
            return self.is_done()

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
