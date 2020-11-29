from AlphaZero_Qttt.main_story import *


class args:
    numIters = 1000
    updateThreshold = 0.55  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    training_dataset_limit = 500000  # Number of game examples to train the neural networks.
    fresh_data_percentage = 0.4 # keep 0.6 percent of stale data from previous iteration
    numMCTSSims = 200  # Number of games moves for MCTS to simulate.
    roundsOfBattle = 40  # Number of games to play during arena play to determine if new net will be accepted.
    cpuct = 1

    # path_checkpoints = '/content/gdrive/My Drive/checkpoints/teamproj/'
    path_checkpoints = 'C:/Users/xiaon/OneDrive/backup/DL/teamProject/my_ttt/test/'
    save_checkpoint_filename = 'team_baseline.pt'
    load_checkpoint_filename = 'team_baseline.pt'
    load_model = False
    numItersForTrainExamplesHistory = 20


if __name__ == '__main__':
    resumeFlag = False
    net = Network()
    if args.load_model:
        print("------------------------Resuming Training from Checkpoint--------------------------")
        net.load_model(args.path_checkpoints, args.load_checkpoint_filename)

    print("------------------------Start Learning--------------------------")
    learn_from_self_play(net, args)
