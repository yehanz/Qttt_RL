from AlphaZero_Qttt.main_story import *


class args:
    batch_size = 512
    weight_decay = 1e-6
    train_epoch = 21
    learning_rate = 1.5e-3

    roundsOfBattle = 50
    numMCTSSims = 100  # Number of games moves for MCTS to simulate.
    cpuct = 1

    # path_checkpoints = '/content/gdrive/My Drive/checkpoints/teamproj/'
    path_checkpoints = 'D:/h4p2_ckp/'
    load_checkpoint_filename1 = 'sym.pt'
    load_checkpoint_filename2 = 'shared_mcts.pt'


if __name__ == '__main__':
    n1 = Network(args)
    n2 = Network(args)

    n1.load_model(args.path_checkpoints, args.load_checkpoint_filename2)
    n1.apply_trans = True
    n2.load_model(args.path_checkpoints, args.load_checkpoint_filename1)
    n2.apply_trans = True

    print(battle(n1, n2, args))
