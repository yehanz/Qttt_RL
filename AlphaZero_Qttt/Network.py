from AlphaZero_Qttt.env_bridge import *

class Network:
    def __init__(self, net, args):
        self.net = net
        self.args = args

    def predict(self, game_env: EnvForDeepRL):
        """

        :param game_env: game env for evaluation
        :return:
            ndarray(72,) action_prob: all invalid action is masked to 0
            state_value(int): state_value of current env
        """
        output = self.net(game_env.qttt.to_tensor()).numpy()

        action_prob = output[:-2]*game_env.valid_action_mask
        state_value = output[-1]
        return action_prob, state_value

    def train(self, training_example):
        """
        normal routine,
        element of training example:(qttt, action_prob, reward)
        use qttt.to_tensor() to convert qttt state to a tensor
        :param training_example:
        :return:
        """

        # first use qttt.to_tensor convert all qttt to tensor before loading
        # to the dataset.
        pass
