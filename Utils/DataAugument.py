import torch
import numpy as np

def rotate(position: list, degree: int):
    rotated_position = np.rot90(np.array(position).reshape((3, 3)), degree // 90, [0, 1]).reshape(-1).tolist()
    return rotated_position

def transpose(position: list):
    transposed_position = np.transpose(np.array(position).reshape((3, 3))).reshape(-1).tolist()
    return transposed_position

'''
def flip(positions: set):
    flipped_positions = set()
    for pos in positions:
        # rotate 90, 180, 270 degree
        for i in range(2):
            new_pos = np.array(pos).reshape((3, 3))
            new_pos = np.flip(new_pos, i)
            flipped_positions.add(tuple(new_pos.reshape(-1).tolist()))
    return flipped_positions
'''

def get_transformations():
    start_pos = tuple(range(9))
    transformations = []
    for i in range(4):
        # rotate
        cur_rotated_transformation = rotate(start_pos, i * 90)
        transformations.append(cur_rotated_transformation)
    reverse_transformations = transformations[:]
    reverse_transformations[1], reverse_transformations[3] = reverse_transformations[3], reverse_transformations[1]

    for i in range(4):
        # rotate + transpose
        cur_pos = transformations[i]
        cur_transposed_transformation = transpose(cur_pos)
        transformations.append(cur_transposed_transformation)
        reverse_transformations.append(rotate(transpose(start_pos), (4 - i) * 90))
    return transformations, reverse_transformations

def get_mapping_sequence():
    mapping_sequence = {}
    key_num = 0
    for i in range(9):
        for j in range(i + 1, 9):
            mapping_sequence[(i, j)] = key_num
            key_num += 1

    transformations, reverse_transformations = get_transformations()
    sequences = []
    reversed_sequences = []
    for i in range(len(transformations)):
        transformation = transformations[i]
        reverse_transformation = reverse_transformations[i]
        cur_sequence = []
        reversed_sequence = []
        for i in range(9):
            for j in range(i + 1, 9):
                x = transformation[i]
                y = transformation[j]
                if x > y :
                    x, y = y, x
                cur_sequence.append(mapping_sequence[(x, y)])
                rx = reverse_transformation[i]
                ry = reverse_transformation[j]
                if rx > ry :
                    rx, ry = ry, rx
                reversed_sequence.append(mapping_sequence[(rx, ry)])
        sequences.append(cur_sequence)
        reversed_sequences.append(reversed_sequence)
    return sequences, reversed_sequences

print(get_transformations()[0])
print(get_transformations()[1])
print(get_mapping_sequence()[0])
print(get_mapping_sequence()[1])

