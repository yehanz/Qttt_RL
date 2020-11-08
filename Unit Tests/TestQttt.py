import numpy as np

from env import Qttt


# Test Qttt
def test_step():
    qttt = Qttt()
    qttt.step((1, 0), 1)
    qttt.step((1, 0), 2)
    assert (qttt.board[0].entangled_marks == [1, 2])
    assert (qttt.board[0].entangled_blocks == [1, 1])

    assert (qttt.board[1].entangled_marks == [1, 2])
    assert (qttt.board[1].entangled_blocks == [0, 0])

    for i in range(2, 9):
        assert (qttt.board[i].entangled_marks == [])
        assert (qttt.board[i].entangled_blocks == [])


def test_has_cycle():
    # one circle with two element
    qttt = Qttt()
    qttt.step((2, 3), 1)

    assert (not qttt.has_cycle)
    qttt.step((2, 3), 2)
    assert qttt.has_cycle

    # one circle with four elements
    qttt2 = Qttt()
    qttt2.step((0, 1), 1)
    assert (not qttt2.has_cycle)
    qttt2.step((1, 2), 2)
    assert (not qttt2.has_cycle)
    qttt2.step((2, 3), 3)
    assert (not qttt2.has_cycle)
    qttt2.step((0, 3), 4)
    assert qttt2.has_cycle

    # test no cycle
    qttt = Qttt()
    qttt.step((0, 1), 1)
    assert (not qttt.has_cycle)
    qttt.step((2, 3), 2)
    assert (not qttt.has_cycle)

    # test no cycle
    qttt = Qttt()
    qttt.step((2, 4), 1)
    assert (not qttt.has_cycle)
    qttt.step((1, 4), 2)
    assert (not qttt.has_cycle)


def test_get_all_possible_collapse():
    # one circle with four elements
    qttt = Qttt()
    qttt.step((0, 1), 1)
    qttt.step((1, 2), 2)
    qttt.step((2, 3), 3)
    qttt.step((0, 3), 4)
    qttt1, qttt2 = qttt.get_all_possible_collapse((0, 3), 4)

    # check qttt1 elements
    expectation_marks = [4, 1, 2, 3]
    for i in range(4):
        assert (qttt1.board[i].entangled_blocks == [])
        assert (qttt1.board[i].entangled_marks == [expectation_marks[i]])

    # check qttt2 elements
    expectation_marks = [1, 2, 3, 4]
    for i in range(4):
        assert (qttt2.board[i].entangled_blocks == [])
        assert (qttt2.board[i].entangled_marks == [expectation_marks[i]])

    # test to propagate to ttt
    qttt1.propagate_qttt_to_ttt()
    assert ((qttt1.ttt.board == np.array([4, 1, 2, 3, -1, -1, -1, -1, -1])).all())

    qttt2.propagate_qttt_to_ttt()
    assert ((qttt2.ttt.board == np.array([1, 2, 3, 4, -1, -1, -1, -1, -1])).all())

    '''
    take a look at special example
            3   2       1,9

            4   1,2,3   8
                4,5,6
                7,8,9

            5   6       7
    '''
    s_qttt = Qttt()
    s_qttt.step((2, 4), 1)
    s_qttt.step((1, 4), 2)
    s_qttt.step((0, 4), 3)
    s_qttt.step((3, 4), 4)
    s_qttt.step((6, 4), 5)
    s_qttt.step((7, 4), 6)
    s_qttt.step((8, 4), 7)
    s_qttt.step((5, 4), 8)
    s_qttt.step((2, 4), 9)
    # s_qttt.visualize_board()

    qttt1, qttt2 = s_qttt.get_all_possible_collapse((2, 4), 9)
    '''
    Finally, we get 2 possible collapsed Qttt state:
            3   2   9       3   2   1

            4   1   8   or  4   9   8

            5   6   7       5   6   7
    '''

    # check qttt1 elements
    expectation_marks = [3, 2, 9, 4, 1, 8, 5, 6, 7]
    for i in range(4):
        assert (qttt1.board[i].entangled_blocks == [])
        assert (qttt1.board[i].entangled_marks == [expectation_marks[i]])

    # check qttt2 elements
    expectation_marks = [3, 2, 1, 4, 9, 8, 5, 6, 7]
    for i in range(4):
        assert (qttt2.board[i].entangled_blocks == [])
        assert (qttt2.board[i].entangled_marks == [expectation_marks[i]])

    # test to propagate to ttt
    qttt1.propagate_qttt_to_ttt()
    # qttt1.ttt.visualize_board()
    assert ((qttt1.ttt.board == np.array([3, 2, 9, 4, 1, 8, 5, 6, 7])).all())

    qttt2.propagate_qttt_to_ttt()
    assert ((qttt2.ttt.board == np.array([3, 2, 1, 4, 9, 8, 5, 6, 7])).all())


def test_get_free_QBlock_ids():
    # one circle with four elements
    qttt = Qttt()
    qttt.step((0, 1), 1)
    qttt.step((1, 2), 2)
    qttt.step((2, 3), 3)
    qttt.step((0, 3), 4)
    assert ((qttt.get_free_QBlock_ids() == range(0, 9)).all())
    qttt1, qttt2 = qttt.get_all_possible_collapse((0, 3), 4)
    qttt1.propagate_qttt_to_ttt()
    assert ((qttt1.get_free_QBlock_ids() == range(4, 9)).all())
    qttt2.propagate_qttt_to_ttt()
    assert ((qttt2.get_free_QBlock_ids() == range(4, 9)).all())


test_step()
test_has_cycle()
test_get_all_possible_collapse()
test_get_free_QBlock_ids()
