from env import Env, Qttt
import numpy as np

def test_step():
    # one circle with two element
    env = Env()
    next_qttt = env.step(Qttt(), (1, 0), 1)[0]
    free_qblock_id_lists, collapsed_qttts = env.get_valid_moves()
    assert((free_qblock_id_lists[0] == range(9)).all())

    next_qttt = env.step(next_qttt, (1, 0), 2)[0]
    free_qblock_id_lists, collapsed_qttts = env.get_valid_moves()
    assert((free_qblock_id_lists[0] == range(2, 9)).all())
    assert((free_qblock_id_lists[1] == range(2, 9)).all())
    assert((collapsed_qttts[0].ttt.board[:2] == np.array([1, 2])).all())
    assert((collapsed_qttts[1].ttt.board[:2] == np.array([2, 1])).all())
    assert((collapsed_qttts[0].ttt.board[:2] == np.array([1, 2])).all())
    assert((collapsed_qttts[1].ttt.board[:2] == np.array([2, 1])).all())

    '''
    take a look at special example
            3   2       1,9

            4   1,2,3   8
                4,5,6
                7,8,9

            5   6       7
    '''
    s_env = Env()
    next_qttt = s_env.step(Qttt(), (2, 4), 1)[0]
    next_qttt = s_env.step(next_qttt, (1, 4), 2)[0]
    next_qttt = s_env.step(next_qttt, (0, 4), 3)[0]
    next_qttt = s_env.step(next_qttt, (3, 4), 4)[0]
    next_qttt = s_env.step(next_qttt, (6, 4), 5)[0]
    next_qttt = s_env.step(next_qttt, (7, 4), 6)[0]
    next_qttt = s_env.step(next_qttt, (8, 4), 7)[0]
    next_qttt = s_env.step(next_qttt, (5, 4), 8)[0]
    params = s_env.step(next_qttt, (2, 4), 9)[0]
    print(params)

    # qttt1, qttt2 = s_qttt.get_all_possible_collapse((2, 4), 9)

def test_get_valid_moves():
    pass

test_step()
test_get_valid_moves()