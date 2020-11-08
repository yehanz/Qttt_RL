import numpy as np

from env import Qttt


def test_has_won():
    ttt = Qttt.ttt()
    assert (ttt.has_won() == (False, "NO"))
    ttt.board = np.array([0, 1, -1, -1, -1, -1, -1, -1, -1])
    assert (ttt.has_won() == (False, "NO"))
    ttt.board = np.array([1, 2, -1, 3, 4, -1, 5, 6, -1])
    assert (ttt.has_won() == (True, "XY_WIN"))
    ttt.board = np.array([1, 2, -1, 3, 4, -1, 7, 6, -1])
    assert (ttt.has_won() == (True, "YX_WIN"))
    ttt.board = np.array([3, 2, 9, 4, 1, 8, 5, 6, 7])
    assert (ttt.has_won() == (True, "X_WIN"))
    ttt.board = np.array([1, 3, 2, -1, 4, 6, -1, 8, 5])
    assert (ttt.has_won() == (False, "NO"))
    ttt.board = np.array([1, 3, 2, 7, 4, 6, -1, 8, 5])
    assert (ttt.has_won() == (True, "TIE"))
    ttt.board = np.array([3, 2, 1, 4, 9, 8, 5, 6, 7])
    assert (ttt.has_won() == (True, "X_WIN"))
    ttt.board = np.array([2, 4, 6, -1, -1, -1, -1, -1, -1])
    assert (ttt.has_won() == (True, "Y_WIN"))


test_has_won()
