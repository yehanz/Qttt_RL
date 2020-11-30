from functools import partial

from AlphaZero_Qttt.env_bridge import *
from AlphaZero_Qttt.main_story import *


def one_round_of_battle(net, config, human_first):
    game_env = EnvForDeepRL()
    game_env.qttt.visualize_board()
    mc = MCTS(EnvForDeepRL(), net, config.numMCTSSims, config.cpuct)
    human_actor = partial(get_action_from_human, game_env)
    machine_actor = partial(get_action_from_machine, mc, game_env)
    # actors = [human_actor, machine_actor]
    actors = [human_actor, human_actor]
    if not human_first:
        actors.reverse()

    while True:
        actor = actors[game_env.current_player_id]
        action_code = actor()
        # step action
        _, _, reward, done = game_env.act(action_code)
        mc.step(action_code)
        game_env.qttt.visualize_board()
        if done:
            if reward > 0:
                winner = 'human' if actor is human_actor else 'machine'
                print('winner ' + winner)
            elif reward < 0:
                winner = 'machine' if actor is human_actor else 'human'
                print('winner ' + winner)
            else:
                winner = 'tie'
                print('no winner!')
            return winner


def get_action_from_machine(mc, game_env):
    # set temperature to 0 to get the strongest possible move
    _, policy_given_state = mc.get_action_prob(temp=0)
    action_code = game_env.pick_a_valid_move(policy_given_state, is_train=False)
    return action_code


def get_action_from_human(game_env: EnvForDeepRL):
    choice = 0
    if game_env.qttt.has_cycle:
        game_env.collapsed_qttts[0].visualize_board()
        game_env.collapsed_qttts[1].visualize_board()
        choice = int(input('we got a cycle, please choose a collapse version: 0 or 1?'))
        print('You chose this one:')
        game_env.collapsed_qttts[choice].visualize_board()
        if game_env.collapsed_qttts[choice].has_won()[0]:
            return 72 + choice

    print('valid block ids ' + str(game_env.next_valid_moves[choice]))
    block_ids = input('please drop a piece, input 2 unique numbers ranging 0-8, like \'07\'')
    id_num = sorted([int(x) for x in block_ids])
    if len(id_num) != 2 or id_num[0] not in game_env.next_valid_moves[choice] or id_num[1] not in \
            game_env.next_valid_moves[choice]:
        print('invalid move!')
        exit(0)

    action_code = 36 * choice + MOVE_TO_INDEX[id_num[0]][id_num[1]]
    return action_code


class args:
    numMCTSSims = 50  # Number of games moves for MCTS to simulate.
    cpuct = 1

    # path_checkpoints = '/content/gdrive/My Drive/checkpoints/teamproj/'
    path_checkpoints = 'D:/h4p2_ckp/'
    load_checkpoint_filename = 'team_baseline2.pt'


if __name__ == '__main__':
    net = Network()
    print("------------------------Load Model--------------------------")
    net.load_model(args.path_checkpoints, args.load_checkpoint_filename)

    score_board = {'machine': 0, 'human': 0, 'tie': 0}

    mod = input('1 for human play first, 2 for machine play first, 0 to get score borad summary and exit\n')
    mod = int(mod)
    while mod != 0:
        score_board[one_round_of_battle(net, args, mod == 1)] += 1
        mod = int(input('1 for human play first, 2 for machine play first, 0 to get score borad summary and exit\n'))

    print('score board summary:')
    print(score_board)
