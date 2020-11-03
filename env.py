import numpy as np
from copy import deepcopy

# X is the first player
REWARD = {
    'NO_REWARD': 0.0,
    'Y_WIN_REWARD': 1.0,
    'X_WIN_REWARD': -1.0,
    # both O and X wins, but O wins earlier
    'YX_WIN_REWARD': 0.7,
    # both O and X wins, but X wins earlier
    'XY_WIN_REWARD': -0.7,
    'TIE_REWARD': 0.5,
}


class Env:
    def __init__(self):
        self.qttt = Qttt()
        self.round_ctr = 1

        # all possible collapsed version of current qttt
        self.collapsed_qttts = [Qttt(), Qttt()]
        # for each collapsed version, what's the valid move(ids of free Qlock)
        self.next_valid_moves = [[i for i in range(9)], [i for i in range(9)]]
        # for each collapsed move, what's the block number that we choose for the
        # lastest piece on the board to collapse
        self.collapsed_actions = []

    @property
    def player_id_for_current_round(self):
        # either 0 or 1
        return self.round_ctr % 2

    def get_state_from_constant_view(self):
        """
        if current player play odd piece, return a copy of qttt directly
        else all piece number on current board decrease by 1 so that we
        always get a constant view from player who always play odd pieces

        player with id 0 always play pieces with odd number, with id 1 play
        even number pieces

        :return:
            qttt curr_qttt: constant view of the board
            int whose_turn: id of the current player, which is round_ctr%2
        """
        pass

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

    def get_action_space(self):
        return self.collapsed_actions, self.next_valid_moves

    def step_with_code(self, collapsed_qttt_idx, agent_move):
        self.step(self.collapsed_qttts[collapsed_qttt_idx], agent_move)

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
        done, winner = self.qttt.has_won()
        reward = REWARD[winner + '_REWARD']
        if done:
            # update reward here
            return self.qttt, self.round_ctr, reward, done

        self.qttt.step(agent_move, mark)

        if self.qttt.has_cycle():
            self.collapsed_actions, self.collapsed_qttts = \
                self.qttt.get_all_possible_collapse(agent_move, mark)

        else:
            qttt_copy = deepcopy(self.qttt)
            # always take a list of 2 qttts for convenience
            self.collapsed_qttts = [qttt_copy, qttt_copy]
            self.collapsed_actions = []

        self.next_valid_moves = []
        for qttt in self.collapsed_qttts:
            self.next_valid_moves.append(None if qttt.has_won()[0] else qttt.get_free_QBlock_ids())

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

    def get_state(self):
        """
        State should be read only!
        :return:
        """
        return self.board

    def has_cycle(self):
        def get_graph_info(board):
            node_num = 0
            nodes = set()
            edges = []
            for block in board:
                if not block.mark and block.entangled_blocks:
                    node_num += 1
                    for node1 in block.entangled_blocks:
                        node2 = block.block_id
                        nodes.add(node1)
                        nodes.add(node2)
                        if node1 > node2:
                            continue
                        edges.append((node1, node2))
            return node_num, nodes, edges

        def valid_tree(n, edges):
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

        node_num, nodes, edges = get_graph_info(self.board)
        # map nodes in edges to (0, node_num)
        mapping = {}
        nodes = list(nodes)
        for i in range(len(nodes)):
            mapping[nodes[i]] = i
        mapped_edges = [[mapping[edge[0]], mapping[edge[1]]] for edge in edges]

        return not valid_tree(node_num, mapped_edges)

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
            list(int): collapsed choice, element represents the block id we choose
                    to collapse as the final block for the spooky piece the
                    opponent placed last time.
        """

        def consequent_collapse(board, collapsed_block_id, last_mark):
            collapsed_block = board[collapsed_block_id]
            if collapsed_block.mark is not None:
                return
            entangled_block_ids, entangled_marks = collapsed_block.collapse(last_mark)
            for i in range(len(entangled_block_ids)):
                consequent_collapse(board, entangled_block_ids[i], entangled_marks[i])

        choice1, choice2 = last_move

        possible_collapse1 = deepcopy(self)
        consequent_collapse(possible_collapse1.board, choice1, last_mark)
        # always update corresponding ttt state
        possible_collapse1.propagate_qttt_to_ttt()

        possible_collapse2 = deepcopy(self)
        consequent_collapse(possible_collapse2.board, choice2, last_mark)
        # always update corresponding ttt state
        possible_collapse2.propagate_qttt_to_ttt()

        possible_collapse = [possible_collapse1, possible_collapse2]
        return [choice1, choice2], possible_collapse

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

    def visualize_board(self):
        # visualize the Qttt board
        for i in range(3):
            print("{:9s}|{:9s}|{:9s}".format(
                *[" ".join([str(integer) for integer in self.board[k].entangled_marks]) for k in
                  range(i * 3, i * 3 + 3)]))

    def propagate_qttt_to_ttt(self):
        """
        update ttt state with Qttt state, where a block of ttt is occupied by some mark
        only if the corresponding QBlock has collapsed

        :return:
            None
        """
        # update ttt board
        for i in range(9):
            self.ttt.board[i] = self.board[i].mark if self.board[i].mark else 0

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
                    if moves[idx] == 0:
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
                X_win = (3 in rows) or (3 in cols) or diag == 3 or disdiag == 3
                Y_win = (-3 in rows) or (-3 in cols) or diag == -3 or disdiag == -3
                if X_win and Y_win:
                    if max_x < max_y:
                        winner = 'XY_WIN'
                    else:
                        winner = 'YX_WIN'
                elif X_win:
                    winner = 'X_WIN'
                elif Y_win:
                    winner = 'Y_WIN'
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
            return np.where(self.board == 0)[0]

        def visualize_board(self):
            # visualize the ttt board
            for i in range(3):
                print("{:2d}|{:2d}|{:2d}".format(*[self.board[k] for k in range(i * 3, i * 3 + 3)]))
