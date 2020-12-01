from Utils.DataAugument import rotate, transpose

def test_rotate():
    start_pos = tuple(range(9))
    rotated_transformation = rotate(start_pos, 90)
    assert(rotated_transformation == [2, 5, 8, 1, 4, 7, 0, 3, 6])
    rotated_transformation = rotate(start_pos, 180)
    assert(rotated_transformation == [8, 7, 6, 5, 4, 3, 2, 1, 0])
    rotated_transformation = rotate(start_pos, 270)
    assert(rotated_transformation == [6, 3, 0, 7, 4, 1, 8, 5, 2])
    rotated_transformation = rotate(start_pos, 360)
    assert(rotated_transformation == [0, 1, 2, 3, 4, 5, 6, 7, 8])

def test_transpose():
    start_pos = tuple(range(9))
    transposed_transformation = transpose(start_pos)
    print(transposed_transformation)
    assert(transposed_transformation == [0, 3, 6, 1, 4, 7, 2, 5, 8])

'''
def test_flip():
    start_pos = tuple(range(9))
    transformations = set()
    transformations.add(start_pos)
    # step 1 operate rotation
    flipped_transformations = flip(transformations)
    assert (len(flipped_transformations) == 2)
    assert (flipped_transformations == {(2, 1, 0, 5, 4, 3, 8, 7, 6), (6, 7, 8, 3, 4, 5, 0, 1, 2)})
'''

test_rotate()
test_transpose()


