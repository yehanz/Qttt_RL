import numpy as np

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
    assert ((collapsed_qttts[0].ttt.board == (np.ones(9) * (-1))).all())

    params = s_env.step(s_env.get_valid_moves()[1][0], (2, 4))
    free_qblock_id_lists, collapsed_qttts, _ = s_env.get_valid_moves()
    assert (not free_qblock_id_lists[0])
    assert (not free_qblock_id_lists[1])
    assert ((collapsed_qttts[0].ttt.board == np.array([3, 2, 9, 4, 1, 8, 5, 6, 7])).all())
    assert ((collapsed_qttts[1].ttt.board == np.array([3, 2, 1, 4, 9, 8, 5, 6, 7])).all())
    assert (params[1] == 10)
    assert (params[2] == 0)
    assert (not params[3])

    params = s_env.step(collapsed_qttts[1], (2, 4), 9)
    assert (params[2] == -1)
    assert (params[3])


test_step()
