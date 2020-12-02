import numpy as np
import torch

INDEX_TO_MOVE = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
    (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 5), (4, 6), (4, 7), (4, 8),
    (5, 6), (5, 7), (5, 8),
    (6, 7), (6, 8),
    (7, 8),
]

# test case of a smaller board
INDEX_TO_MOVE2 = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3),
    (2, 3),
]


def rot0(board):
    return board


def rot90(board):
    return board.rot90(1, [-2, -1])


def rot180(board):
    return board.rot90(2, [-2, -1])


def rot270(board):
    return board.rot90(3, [-2, -1])


def T(board):
    return board.transpose(-2, -1)


def rot0T(board):
    return board.transpose(-2, -1)


def rot90T(board):
    return T(rot90(board))


def rot180T(board):
    return T(rot180(board))


def rot270T(board):
    return T(rot270(board))


def get_board_id_mapping_with_board_trans_func(l):
    initial = torch.arange(l ** 2).reshape(1, l, l)
    res = []
    for trans in board_transforms:
        res.append(trans(initial).reshape(-1))
    return res


def get_i2m_mapping(board_maps, i2m_):
    prob_vector_mappings = []
    for board_map in board_maps:
        prob_action_map = []
        prob_vec_mapping = []
        for i in i2m_:
            new_i = tuple(sorted((board_map[i[0]].item(), board_map[i[1]].item())))
            prob_action_map.append(new_i)
        for i in prob_action_map:
            prob_vec_mapping.append(i2m_.index(i))
        prob_vector_mappings.append(prob_vec_mapping)
    return prob_vector_mappings


def i2m_to_prob_vec_trans(single_index2move_transforms):
    i2m_trans = np.array(single_index2move_transforms)
    prob_vec_trans = np.zeros((i2m_trans.shape[0], i2m_trans.shape[1] * 2 + 2), dtype=int)
    prob_vec_trans[:, :36] = i2m_trans
    prob_vec_trans[:, 36:72] = i2m_trans + 36
    prob_vec_trans[:, -2] = 36 * 2
    prob_vec_trans[:, -1] = 36 * 2 + 1
    return prob_vec_trans


board_transforms = [rot0, rot90, rot180, rot270, rot0T, rot90T, rot180T, rot270T]
single_index2move_transforms = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31, 32, 33, 34, 35],
    [17, 20, 8, 16, 19, 1, 15, 18, 32, 11, 26, 31, 4, 22, 30, 14, 29, 35, 7, 25, 34, 10, 13, 0, 9, 12, 28, 3, 21, 27, 6,
     24, 33, 2, 5, 23],
    [35, 34, 32, 29, 25, 20, 14, 7, 33, 31, 28, 24, 19, 13, 6, 30, 27, 23, 18, 12, 5, 26, 22, 17, 11, 4, 21, 16, 10, 3,
     15, 9, 2, 8, 1, 0],
    [23, 5, 33, 27, 12, 34, 30, 18, 2, 24, 21, 9, 25, 22, 15, 6, 3, 0, 7, 4, 1, 28, 13, 35, 31, 19, 10, 29, 26, 16, 14,
     11, 8, 32, 20, 17],
    [2, 5, 0, 3, 6, 1, 4, 7, 23, 9, 21, 24, 15, 22, 25, 12, 27, 33, 18, 30, 34, 10, 13, 8, 11, 14, 28, 16, 26, 29, 19,
     31, 35, 17, 20, 32],
    [8, 1, 17, 16, 15, 20, 19, 18, 0, 11, 10, 9, 14, 13, 12, 4, 3, 2, 7, 6, 5, 26, 22, 32, 31, 30, 21, 29, 28, 27, 25,
     24, 23, 35, 34, 33],
    [32, 20, 35, 29, 14, 34, 25, 7, 17, 31, 26, 11, 30, 22, 4, 19, 16, 8, 18, 15, 1, 28, 13, 33, 24, 6, 10, 27, 21, 3,
     12, 9, 0, 23, 5, 2],
    [33, 34, 23, 27, 30, 5, 12, 18, 35, 24, 28, 31, 6, 13, 19, 25, 29, 32, 7, 14, 20, 21, 22, 2, 9, 15, 26, 3, 10, 16,
     4, 11, 17, 0, 1, 8]
]
prob_vec_transforms = i2m_to_prob_vec_trans(single_index2move_transforms)


def test_2():
    a = torch.arange(2 * 3 * 3).reshape(2, 3, 3)
    print(a)
    print(board_transforms[4](a))


def test_3():
    a = torch.arange(3)
    b = torch.zeros(4, 8)
    b[:, :3] = a
    b[:, 3:6] = a + 3
    b[:, -2] = 3 * 2
    b[:, -1] = 3 * 2 + 1
    assert (b == torch.arange(8).repeat(4, 1)).all()


def test_relationship_between_transform_and_invert_transform():
    l = 3
    initial_borad = torch.arange(l ** 2).reshape(1, l, l)
    initial_i2m = np.array(single_index2move_transforms[0])
    for i in range(4):
        ted_board = board_transforms[i](initial_borad)
        ted_vec = initial_i2m[single_index2move_transforms[i]]
        print(np.array(initial_borad == board_transforms[(4 - i) % 4](ted_board)).all())
        print(np.array(initial_i2m == ted_vec[single_index2move_transforms[(4 - i) % 4]]).all())

    for i in range(4, 8):
        ted_board = board_transforms[i](initial_borad)
        ted_vec = initial_i2m[single_index2move_transforms[i]]
        print(np.array(initial_borad == board_transforms[i](ted_board)).all())
        print(np.array(initial_i2m == ted_vec[single_index2move_transforms[i]]).all())


if __name__ == '__main__':
    test_relationship_between_transform_and_invert_transform()
