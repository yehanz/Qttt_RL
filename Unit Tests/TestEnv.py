from copy import deepcopy

import numpy as np
import torch

from AlphaZero_Qttt.env_bridge import EnvForDeepRL
from env import Env, Qttt


def test_step():
    # one circle with two element
    env = Env()
    _ = env.step(Qttt(), (1, 0))[0]
    free_qblock_id_lists, collapsed_qttts, collapse_choice = env.get_valid_moves()
    assert (len(free_qblock_id_lists) == 1)
    assert (len(free_qblock_id_lists) == len(collapsed_qttts))
    assert ((free_qblock_id_lists[0] == range(9)).all())
    assert (collapse_choice == ())

    agent_move = (1, 0)
    _ = env.step(collapsed_qttts[0], agent_move)[0]
    free_qblock_id_lists, collapsed_qttts, collapse_choice = env.get_valid_moves()
    assert (collapse_choice == agent_move)
    assert ((free_qblock_id_lists[0] == range(2, 9)).all())
    assert ((free_qblock_id_lists[1] == range(2, 9)).all())
    assert (len(free_qblock_id_lists) == 2)
    assert (len(free_qblock_id_lists) == len(collapsed_qttts))
    assert ((collapsed_qttts[0].ttt.board[:2] == np.array([1, 2])).all())
    assert ((collapsed_qttts[1].ttt.board[:2] == np.array([2, 1])).all())
    assert ((collapsed_qttts[0].ttt.board[:2] == np.array([1, 2])).all())
    assert ((collapsed_qttts[1].ttt.board[:2] == np.array([2, 1])).all())

    _ = env.step(collapsed_qttts[0], (2, 3))[0]
    free_qblock_id_lists, collapsed_qttts, collapse_choice = env.get_valid_moves()
    assert (collapse_choice == ())
    assert (len(free_qblock_id_lists) == 1)
    assert (len(free_qblock_id_lists) == len(collapsed_qttts))
    '''
    take a look at special example
            3   2       1,9

            4   1,2,3   8
                4,5,6
                7,8,9

            5   6       7
    '''
    s_env = Env()
    _ = s_env.step(Qttt(), (2, 4))[0]
    _ = s_env.step(s_env.get_valid_moves()[1][0], (1, 4))[0]
    _ = s_env.step(s_env.get_valid_moves()[1][0], (0, 4))[0]
    _ = s_env.step(s_env.get_valid_moves()[1][0], (3, 4))[0]
    _ = s_env.step(s_env.get_valid_moves()[1][0], (6, 4))[0]
    _ = s_env.step(s_env.get_valid_moves()[1][0], (7, 4))[0]
    _ = s_env.step(s_env.get_valid_moves()[1][0], (8, 4))[0]
    _ = s_env.step(s_env.get_valid_moves()[1][0], (5, 4))[0]
    free_qblock_id_lists, collapsed_qttts, _ = s_env.get_valid_moves()
    assert (len(free_qblock_id_lists) == 1)
    assert (len(free_qblock_id_lists) == len(collapsed_qttts))
    assert ((free_qblock_id_lists[0] == range(9)).all())
    assert ((collapsed_qttts[0].ttt.board == (np.ones(9) * Qttt.ttt.EMPTY_BLOCK_VAL)).all())

    params = s_env.step(s_env.get_valid_moves()[1][0], (2, 4))
    free_qblock_id_lists, collapsed_qttts, _ = s_env.get_valid_moves()
    assert (not free_qblock_id_lists[0])
    assert (not free_qblock_id_lists[1])
    assert ((collapsed_qttts[0].ttt.board == np.array([3, 2, 9, 4, 1, 8, 5, 6, 7])).all())
    assert ((collapsed_qttts[1].ttt.board == np.array([3, 2, 1, 4, 9, 8, 5, 6, 7])).all())
    assert (params[1] == 10)
    assert (params[2] == 0)
    assert (not params[3])

    params = s_env.step(collapsed_qttts[1], None)
    assert (params[2] == -1)
    assert (params[3])

# in normal view, we have (r+p) % 2 = 1
# in even piece view, we have (r+p) % 2 = 0
def test_view_of_env1():
    # r 1 p 0
    s_env = EnvForDeepRL()
    # r 2 p 1
    _ = s_env.step(Qttt(), (0, 1))[0]
    # r 3 p 0
    _ = s_env.step(s_env.collapsed_qttts[0], (0, 1))[0]

    # change when it has already been normal view
    before_view_change = deepcopy(s_env)
    s_env.change_to_normal_view()
    # r 3 p 0
    assert before_view_change == s_env

    # odd piece round change to even view, everything except player id +1
    # r 4 p 0
    s_env.change_to_even_pieces_view()
    _, collapsed_qttts, _ = s_env.get_valid_moves()
    assert s_env.round_ctr == 4
    assert s_env.current_player_id == 0
    assert (collapsed_qttts[1].ttt.board == np.array([2, 3, 0, 0, 0, 0, 0, 0, 0])).all()

    # in converted even piece view, we continue to play
    # r 5 p 1
    _ = s_env.step(s_env.collapsed_qttts[1], (3, 4))[0]
    assert s_env.round_ctr == 5
    assert s_env.current_player_id == 1
    assert (s_env.collapsed_qttts[0].ttt.board == np.array([2, 3, 0, 0, 0, 0, 0, 0, 0])).all()
    assert s_env.collapsed_qttts[0].board[3].entangled_blocks == [4]
    assert s_env.collapsed_qttts[0].board[3].entangled_marks == [4]

    # in odd piece round of converted view env, it will back to normal view, even piece round
    # r 4 p 1
    s_env.change_to_normal_view()
    before_view_change = deepcopy(s_env)
    assert s_env.round_ctr == 4
    assert s_env.current_player_id == 1
    assert (s_env.qttt.ttt.board == np.array([1, 2, 0, 0, 0, 0, 0, 0, 0])).all()
    assert s_env.qttt.board[3].entangled_blocks == [4]
    assert s_env.qttt.board[3].entangled_marks == [3]
    s_env.change_to_even_pieces_view()
    assert s_env == before_view_change


def test_view_of_env2():
    # r 1 p 0
    s_env = EnvForDeepRL()
    # r 2 p 1
    s_env.step(Qttt(), (0, 4))
    # r 3 p 0
    s_env.step(s_env.get_valid_moves()[1][0], (1, 4))
    # r 4 p 1
    s_env.step(s_env.get_valid_moves()[1][0], (2, 4))
    # r 5 p 0
    s_env.step(s_env.get_valid_moves()[1][0], (3, 4))
    # r 6 p 1
    s_env.step(s_env.get_valid_moves()[1][0], (5, 4))
    # r 7 p 0
    s_env.step(s_env.get_valid_moves()[1][0], (6, 4))
    # r 8 p 1
    s_env.step(s_env.get_valid_moves()[1][0], (7, 4))

    assert s_env.player_id == 1
    assert s_env.round_ctr == 8
    before_view_change = deepcopy(s_env)
    # r 8 p 1
    s_env.change_to_even_pieces_view()

    # during even piece round nothing should change
    assert before_view_change.player_id == s_env.player_id
    assert before_view_change.round_ctr == s_env.round_ctr
    assert before_view_change.qttt == s_env.qttt
    assert before_view_change.collapse_choice == s_env.collapse_choice

    # r 9 p 0
    _ = s_env.step(s_env.get_valid_moves()[1][0], (8, 4))[0]
    assert s_env.player_id == 0
    assert s_env.round_ctr == 9
    # round for odd piece
    # r 10 p 0
    s_env.change_to_even_pieces_view()

    assert s_env.player_id == 0
    assert s_env.round_ctr == 10
    _, collapsed_qttts, _ = s_env.get_valid_moves()
    for i in range(3):
        assert collapsed_qttts[0].board[i].entangled_marks == [i + 2]
        assert collapsed_qttts[0].board[i].entangled_blocks == [4]
    for i in range(5, 9):
        assert collapsed_qttts[0].board[i].entangled_marks == [i + 1]
        assert collapsed_qttts[0].board[i].entangled_blocks == [4]
    assert collapsed_qttts[0] == s_env.qttt
    assert collapsed_qttts[0].board[4].entangled_blocks == [0, 1, 2, 3, 5, 6, 7, 8]
    assert collapsed_qttts[0].board[4].entangled_marks == [2, 3, 4, 5, 6, 7, 8, 9]
    assert (collapsed_qttts[0].ttt.board == np.ones(9) * Qttt.ttt.EMPTY_BLOCK_VAL).all()
    # r 9 p 0
    s_env.change_to_normal_view()
    assert s_env.player_id == 0
    assert s_env.round_ctr == 9
    _, collapsed_qttts, _ = s_env.get_valid_moves()
    for i in range(3):
        assert collapsed_qttts[0].board[i].entangled_marks == [i + 1]
        assert collapsed_qttts[0].board[i].entangled_blocks == [4]
    for i in range(5, 9):
        assert collapsed_qttts[0].board[i].entangled_marks == [i]
        assert collapsed_qttts[0].board[i].entangled_blocks == [4]

    assert collapsed_qttts[0].board[4].entangled_blocks == [0, 1, 2, 3, 5, 6, 7, 8]
    assert collapsed_qttts[0].board[4].entangled_marks == [1, 2, 3, 4, 5, 6, 7, 8]
    assert (collapsed_qttts[0].ttt.board == np.ones(9) * Qttt.ttt.EMPTY_BLOCK_VAL).all()

    # r 10 p 1
    s_env.step(s_env.get_valid_moves()[1][0], [0, 4])
    _, collapsed_qttts, _ = s_env.get_valid_moves()
    for i in range(3):
        assert collapsed_qttts[1].board[i].entangled_marks == [i + 1]
        assert collapsed_qttts[1].board[i].entangled_blocks == []
    for i in range(5, 9):
        assert collapsed_qttts[1].board[i].entangled_marks == [i]
        assert collapsed_qttts[1].board[i].entangled_blocks == []
    assert collapsed_qttts[1].board[4].entangled_marks == [9]
    assert collapsed_qttts[1].board[4].entangled_blocks == []
    assert (collapsed_qttts[1].ttt.board == np.array([1, 2, 3, 4, 9, 5, 6, 7, 8])).all()

    # r 10 p 1
    s_env.change_to_even_pieces_view()
    _, collapsed_qttts, _ = s_env.get_valid_moves()
    for i in range(3):
        assert collapsed_qttts[1].board[i].entangled_marks == [i + 1]
        assert collapsed_qttts[1].board[i].entangled_blocks == []
    for i in range(5, 9):
        assert collapsed_qttts[1].board[i].entangled_marks == [i]
        assert collapsed_qttts[1].board[i].entangled_blocks == []
    assert collapsed_qttts[1].board[4].entangled_marks == [9]
    assert collapsed_qttts[1].board[4].entangled_blocks == []
    assert (collapsed_qttts[1].ttt.board == np.array([1, 2, 3, 4, 9, 5, 6, 7, 8])).all()


def test_to_tensor():
    s_env = EnvForDeepRL()
    result_tensor = torch.zeros(11, 9)

    _ = s_env.step(Qttt(), (0, 4))[0]
    result_tensor[1, [0, 4]] = 1
    _ = s_env.step(s_env.get_valid_moves()[1][0], (1, 4))[0]
    result_tensor[2, [1, 4]] = 1
    _ = s_env.step(s_env.get_valid_moves()[1][0], (2, 4))[0]
    result_tensor[3, [2, 4]] = 1
    _ = s_env.step(s_env.get_valid_moves()[1][0], (3, 4))[0]
    result_tensor[4, [3, 4]] = 1
    _ = s_env.step(s_env.get_valid_moves()[1][0], (5, 4))[0]
    result_tensor[5, [5, 4]] = 1
    _ = s_env.step(s_env.get_valid_moves()[1][0], (6, 4))[0]
    result_tensor[6, [6, 4]] = 1
    _ = s_env.step(s_env.get_valid_moves()[1][0], (7, 4))[0]
    result_tensor[7, [7, 4]] = 1
    _ = s_env.step(s_env.get_valid_moves()[1][0], (8, 4))[0]
    result_tensor[8, [8, 4]] = 1
    assert (result_tensor.reshape(11, 3, 3) == s_env.qttt.to_tensor()).all()

    _ = s_env.step(s_env.get_valid_moves()[1][0], (0, 4))[0]

    result_tensor = torch.zeros(11, 9)
    result_tensor[1, 0] = 1
    result_tensor[2, 1] = 1
    result_tensor[3, 2] = 1
    result_tensor[4, 3] = 1
    result_tensor[5, 5] = 1
    result_tensor[6, 6] = 1
    result_tensor[7, 7] = 1
    result_tensor[8, 8] = 1
    result_tensor[9, 4] = 1
    result_tensor[0][:] = 1
    # update collapsed_qttt even when it's done!
    s_env.step(s_env.collapsed_qttts[1], None)
    assert (result_tensor.reshape(11, 3, 3) == s_env.qttt.to_tensor()).all()

    s_env.change_to_even_pieces_view()
    result_tensor = torch.zeros(11, 9)
    result_tensor[2, 0] = 1
    result_tensor[3, 1] = 1
    result_tensor[4, 2] = 1
    result_tensor[5, 3] = 1
    result_tensor[6, 5] = 1
    result_tensor[7, 6] = 1
    result_tensor[8, 7] = 1
    result_tensor[9, 8] = 1
    result_tensor[10, 4] = 1
    result_tensor[0][:] = 1
    assert (result_tensor.reshape(11, 3, 3) == s_env.qttt.to_tensor()).all()


test_step()
test_view_of_env1()
test_view_of_env2()
test_to_tensor()
