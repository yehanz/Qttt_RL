from env import Qttt

# Test Qttt
def test_step():
    qttt = Qttt()
    qttt.step((1,0), 1)
    qttt.step((1,0), 2)
    assert(qttt.board[0].entangled_marks == [1, 2])
    assert(qttt.board[0].entangled_blocks == [1, 1])
    assert(qttt.board[1].entangled_marks == [1, 2])
    assert(qttt.board[1].entangled_blocks == [0, 0])

def test_has_cycle():
    # one circle with two element
    qttt = Qttt()
    qttt.step((1,0), 1)
    assert(qttt.has_cycle() == False)
    qttt.step((1,0), 2)
    assert(qttt.has_cycle() == True)

    # one circle with four elements
    qttt2 = Qttt()
    qttt2.step((0, 1), 1)
    assert(qttt2.has_cycle() == False)
    qttt2.step((1, 2), 2)
    assert (qttt2.has_cycle() == False)
    qttt2.step((2, 3), 3)
    assert (qttt2.has_cycle() == False)
    qttt2.step((0, 3), 4)
    assert (qttt2.has_cycle() == True)

def test_get_all_possible_collapse():
    # one circle with four elements
    qttt = Qttt()
    qttt.step((0, 1), 1)
    qttt.step((1, 2), 2)
    qttt.step((2, 3), 3)
    qttt.step((0, 3), 4)
    qttt.visualize_board()
    qttt1, qttt2 = qttt.get_all_possible_collapse((0, 3), 4)
    qttt1.visualize_board()
    qttt2.visualize_board()



# Test QBlock
def test_place_mark():
    qttt = Qttt()
    block = Qttt.board[1]
    block.entangled_blocks = [2,3]
    block.entangled_marks = [5,6]

test_step()
test_has_cycle()
test_get_all_possible_collapse()
# test_place_mark()