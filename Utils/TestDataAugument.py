from Utils.DataAugument import DataAugument

def test_rotate():
    start_pos = tuple(range(9))
    transformations = set()
    transformations.add(start_pos)
    print(transformations)
    # step 1 operate rotation
    Data = DataAugument()
    rotated_transformations = Data.rotate(transformations)

test_rotate()

