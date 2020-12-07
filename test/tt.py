from Utils.get_sym import *


def record_training_data1(data):
    training_data = []
    # get all symmetric cases of the chess board
    # ATTENTION: when apply board_transforms[i] to qttt tensor, we must apply
    # prob_vec_tranforms[i] correspondingly to the prob vector!
    for d in data:
        s1, s2 = d[0]
        prob = d[1]
        for i in range(len(board_transforms)):
            training_data.append([
                # transformed qttt tensor tuple
                (board_transforms[i](s1), board_transforms[i](s2),),
                # transformed qttt probability vector
                prob[prob_vec_transforms[i]],
                # state value
                d[2]])
    return training_data


if __name__ == '__main__':
    print((-1) ** (0 != 0))
