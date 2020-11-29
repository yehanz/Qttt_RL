from AlphaZero_Qttt.env_bridge import *


def test_valid_action_codes():
    env_bridge = EnvForDeepRL()
    assert (env_bridge.get_valid_action_codes() == np.arange(36 * 2)).all()
    assert (env_bridge.valid_action_mask == np.concatenate((np.ones(36 * 2), np.zeros(2)))).all()
    env_bridge.act(0)
    free_qblock_id_lists, collapsed_qttts, collapse_choice = env_bridge.get_valid_moves()
    assert (len(free_qblock_id_lists) == 2)
    assert ((free_qblock_id_lists[0] == range(9)).all())
    assert (collapse_choice == ())

    env_bridge.act(0)
    free_qblock_id_lists, collapsed_qttts, collapse_choice = env_bridge.get_valid_moves()
    assert (len(free_qblock_id_lists) == 2)
    assert (free_qblock_id_lists[0] == range(2, 9)).all()
    assert ((free_qblock_id_lists[1] == range(2, 9)).all())

    valid_act_code = np.array([MOVE_TO_INDEX[i][j] for i in range(2, 9) for j in range(i + 1, 9)])
    assert (env_bridge.get_valid_action_codes() ==
            np.concatenate((valid_act_code, valid_act_code + 36))).all()

    env_bridge.act(15 + 36)
    env_bridge.act(15 + 36)
    free_qblock_id_lists, collapsed_qttts, collapse_choice = env_bridge.get_valid_moves()
    assert (collapsed_qttts[0].ttt.board == np.array([1, 2, 4, 3, 0, 0, 0, 0, 0])).all()
    assert (collapsed_qttts[1].ttt.board == np.array([1, 2, 3, 4, 0, 0, 0, 0, 0])).all()


def test_all():
    env = EnvForDeepRL()
    env.change_to_even_pieces_view()
    val_act_cod = env.get_valid_action_codes()
    for i in range(8):
        env.act(i)
    assert (env.get_valid_action_codes() == val_act_cod).all()
    # print(env.qttt.to_tensor())
    # print(env.qttt.to_hashable())
    env.act(9)
    # print(env.get_valid_action_codes())
    # print(env.qttt.to_tensor())
    # print(env.qttt.to_tensor())
    # print(env.qttt.to_hashable())
    env.act(72)
    # print(env.qttt.to_hashable())
    # print(env.qttt.to_tensor())
    # print(env.qttt.to_hashable())
    # print(env.qttt.to_tensor())

