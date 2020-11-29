import numpy as np

from env import Qttt


def test_has_won():
    ttt = Qttt.ttt()
    assert (ttt.has_won() == (False, "NO"))
    ttt.board = np.array([0, 1, Qttt.ttt.EMPTY_BLOCK_VAL,
                          Qttt.ttt.EMPTY_BLOCK_VAL, Qttt.ttt.EMPTY_BLOCK_VAL,
                          Qttt.ttt.EMPTY_BLOCK_VAL, Qttt.ttt.EMPTY_BLOCK_VAL,
                          Qttt.ttt.EMPTY_BLOCK_VAL, Qttt.ttt.EMPTY_BLOCK_VAL])
    assert (ttt.has_won() == (False, "NO"))
    ttt.board = np.array([1, 2, Qttt.ttt.EMPTY_BLOCK_VAL, 3, 4,
                          Qttt.ttt.EMPTY_BLOCK_VAL, 5, 6, Qttt.ttt.EMPTY_BLOCK_VAL])
    assert (ttt.has_won() == (True, "ODD_EVEN_WIN"))
    ttt.board = np.array([1, 2, Qttt.ttt.EMPTY_BLOCK_VAL, 3, 4,
                          Qttt.ttt.EMPTY_BLOCK_VAL, 7, 6, Qttt.ttt.EMPTY_BLOCK_VAL])
    assert (ttt.has_won() == (True, "EVEN_ODD_WIN"))
    ttt.board = np.array([3, 2, 9, 4, 1, 8, 5, 6, 7])
    assert (ttt.has_won() == (True, "ODD_WIN"))
    ttt.board = np.array([1, 3, 2, Qttt.ttt.EMPTY_BLOCK_VAL, 4, 6,
                          Qttt.ttt.EMPTY_BLOCK_VAL, 8, 5])
    assert (ttt.has_won() == (False, "NO"))
    ttt.board = np.array([1, 3, 2, 7, 4, 6, Qttt.ttt.EMPTY_BLOCK_VAL, 8, 5])
    assert (ttt.has_won() == (True, "TIE"))
    ttt.board = np.array([3, 2, 1, 4, 9, 8, 5, 6, 7])
    assert (ttt.has_won() == (True, "ODD_WIN"))
    ttt.board = np.array([2, 4, 6, Qttt.ttt.EMPTY_BLOCK_VAL, Qttt.ttt.EMPTY_BLOCK_VAL,
                          Qttt.ttt.EMPTY_BLOCK_VAL, Qttt.ttt.EMPTY_BLOCK_VAL,
                          Qttt.ttt.EMPTY_BLOCK_VAL, Qttt.ttt.EMPTY_BLOCK_VAL])
    assert (ttt.has_won() == (True, "EVEN_WIN"))


test_has_won()
