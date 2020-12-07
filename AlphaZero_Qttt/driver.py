from AlphaZero_Qttt.main_story import *


class args:
    numIters = 1000
    updateThreshold = 0.0  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    training_dataset_limit = 8  # Number of game examples to train the neural networks.
    fresh_data_percentage = 0.3  # keep a given percent of stale data from previous iteration
    numMCTSSims = 5  # Number of games moves for MCTS to simulate.
    roundsOfBattle = 40  # Number of games to play during arena play to determine if new net will be accepted.
    cpuct = 1

    # path_checkpoints = '/content/gdrive/My Drive/checkpoints/teamproj/'
    path_checkpoints = 'C:/Users/xiaon/OneDrive/backup/DL/teamProject/11292241alphaZeroRunnable/test/'
    save_checkpoint_filename = 'deepNN_sym.pt'
    load_checkpoint_filename = 'deepNN_sym.pt'
    training_examples_filename = 'training_example.pt'
    load_model = False
    load_data = False
    skip_initial_data_drop = True

    batch_size = 512
    weight_decay = 1e-6
    train_epoch = 41
    learning_rate = 1.5e-3


if __name__ == '__main__':
    net = Network(args)
    training_examples = []

    # load model ckp and training data ckp if needed
    if args.load_model:
        print("------------------------Resuming Training from Checkpoint--------------------------")
        net.load_model(args.path_checkpoints, args.load_checkpoint_filename)
    if args.load_data:
        training_examples = pickle.load(
            open(args.path_checkpoints + args.training_examples_filename, "rb"))

    print("------------------------Start Learning--------------------------")
    learn_from_self_play(net, args, training_examples)
