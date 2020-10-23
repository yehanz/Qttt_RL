from env import Env, Qttt
import numpy as np

def test_has_won():
    ttt = Qttt.ttt()
    assert (ttt.has_won() == (False, "NO"))
    ttt.board = np.array([1,2,0,3,4,0,5,6,0])
    assert (ttt.has_won() == (True, "XY_WIN"))
    ttt.board = np.array([1,2,0,3,4,0,7,6,0])
    assert (ttt.has_won() == (True, "YX_WIN"))
    ttt.board = np.array([3, 2, 9, 4, 1, 8, 5, 6, 7])
    assert (ttt.has_won() == (True, "X_WIN"))
    ttt.board = np.array([1, 3, 2, 0, 4, 6, 0, 8, 5])
    assert (ttt.has_won() == (False, "NO"))
    ttt.board = np.array([1, 3, 2, 7, 4, 6, 0, 8, 5])
    assert (ttt.has_won() == (True, "TIE"))
    ttt.board = np.array([3, 2, 1, 4, 9, 8, 5, 6, 7])
    assert (ttt.has_won() == (True, "X_WIN"))
    ttt.board = np.array([2, 4, 6, 0, 0, 0, 0, 0, 0])
    assert (ttt.has_won() == (True, "Y_WIN"))

test_has_won()