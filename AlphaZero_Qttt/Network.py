# import sys
# sys.path.append('E:\\CMU_INI\\11785\\project\\Qttt_RL')
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from AlphaZero_Qttt.env_bridge import *
from rl_agent import TD_agent


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=1, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, X):
        out = self.cnn(X)
        return out


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.dimup = False
        if in_channel != out_channel:
            self.dimup = True

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.dimup is True:
            res = self.bn3(self.shortcut(x))
        else:
            res = x
        out += res
        out = F.relu(out)
        return out


# class PolicyHead(nn.Module):
#     def __init__(self, in_channel):
#         super(PolicyHead, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channel, 2, kernel_size=1, stride=1),
#             nn.BatchNorm2d(2),
#             nn.ReLU(),
#         )
#         self.linear = nn.Linear(2, 72)

#     def forward(self, X):
#         out = self.cnn(X)
#         out = self.linear(out)
#         return out.log_softmax()

# class ValueHead(nn.Module):
#     def __init__(self, in_channel):
#         super(ValueHead, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channel, 1, kernel_size=1, stride=1),
#             nn.BatchNorm2d(1),
#             nn.ReLU(),
#         )
#         self.linear1 = nn.Sequential(
#             nn.Linear(1, 512),
#             nn.ReLU(),
#         )
#         self.linear2 = nn.Sequential(
#             nn.Linear(512, 1),
#             nn.Tanh(),
#         )

#     def forward(self, X):
#         out = self.cnn(X)
#         out = self.linear1(out)
#         out = self.linear2(out)
#         return out   


class BasicNetwork(nn.Module):
    def __init__(self):
        super(BasicNetwork, self).__init__()
        self.conv1 = ConvBlock(11, 256)
        self.conv2 = ConvBlock(256, 256)
        self.conv3 = ConvBlock(256, 256)
        self.conv4 = ConvBlock(256, 256)
        self.linear = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        # add 2 more options: only choose the collapsed qttt without dropping any piece
        self.policy_layer = nn.Linear(512, 74)
        self.value_layer = nn.Linear(512, 1)

    def embedding(self, X):
        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        return output

    def forward(self, X, Y):
        X = self.embedding(X)
        Y = self.embedding(Y)
        output = torch.cat((X, Y), dim=1)
        output = output.view(-1, 512 * 3 * 3)
        output = self.linear(output)
        # if prediction, use softmax, if train, use log softmax
        policy = self.policy_layer(output)
        value = torch.tanh(self.value_layer(output))

        return policy, value


class QtttDataset(Dataset):
    def __init__(self, data):
        # data element: ((state1, state2), p, r)
        state_pairs, probs, rewards = zip(*data)
        self.state_pair_tensors = [(state_pair[0].to_tensor(), state_pair[1].to_tensor())
                                   for state_pair in state_pairs]
        self.probs = torch.tensor(probs)
        self.rewards = torch.tensor(rewards)

    def __len__(self):
        return len(self.probs)

    def __getitem__(self, index):
        state_pair = self.state_pair_tensors[index]
        return state_pair[0], state_pair[1], \
               self.probs[index], self.rewards[index]


class Network:
    def __init__(self):
        self.lr = 1e-3
        self.weight_decay = 5e-6
        self.epochs = 10
        self.batch_size = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = BasicNetwork()
        self.net.to(self.device)

    def load_model(self, path_checkpoints, load_checkpoint_filename):
        assert os.path.isdir(path_checkpoints), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(path_checkpoints + load_checkpoint_filename)
        self.net.load_state_dict(checkpoint['model_state_dict'])

    def predict(self, game_env: EnvForDeepRL):
        """

        :param game_env: game env for evaluation
        :return:
            ndarray(72,) action_prob: all invalid action is masked to 0
            state_value(int): state_value of current env
        """
        self.net.eval()
        with torch.no_grad():
            X, Y = game_env.collapsed_qttts[0].to_tensor(), \
                   game_env.collapsed_qttts[1].to_tensor()

            X = X.unsqueeze(0).to(self.device)
            Y = Y.unsqueeze(0).to(self.device)

            output = self.net(X, Y)

        action_prob = np.ones(74) * game_env.valid_action_mask
        # action_prob = F.softmax(output[1], dim=1).squeeze(0).detach().cpu().numpy()[0]
        state_value = 0
        # state_value = output[1].squeeze(0).detach().cpu().numpy()[0]
        return action_prob, state_value

    def train(self, training_example):
        """
        normal routine,
        element of training example:(qttts, action, value)
        use qttt.to_tensor() to convert qttt state to a tensor
        :param training_example:
        :return:
        """

        # first use qttt.to_tensor convert all qttt to tensor before loading
        # to the dataset.

        train_dataset = QtttDataset(training_example)
        train_loader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=0,
                                 pin_memory=True) if torch.cuda.is_available() \
            else dict(shuffle=True, batch_size=2)
        train_loader = DataLoader(train_dataset, **train_loader_args)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.net.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch_idx, (X, Y, target_p, target_v) in enumerate(train_loader):
                optimizer.zero_grad()
                X = X.to(self.device)
                Y = Y.to(self.device)
                target_p = target_p.to(self.device)
                target_v = target_v.to(self.device)

                policy, value = self.net(X, Y)
                policy = F.log_softmax(policy, dim=1)

                loss_p = self.loss_pi(policy, target_p)
                loss_v = self.loss_v(value.view(-1), target_v)

                loss = loss_p + loss_v
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                del X, Y, target_p, target_v

            running_loss /= len(train_loader)
            print('epoch: ', str(epoch + 1), 'Training Loss: ', running_loss)

    def save(self, args):
        torch.save({
            'model_state_dict': self.net.state_dict(),
        },
            args.path_checkpoints + args.save_checkpoint_filename)

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


if __name__ == '__main__':
    network = Network()
    env = EnvForDeepRL()
    agent = TD_agent(1, 0, 1)
    data = []
    for i in range(5):
        policy = np.random.uniform(0, 1, (74,))
        value = np.random.uniform(-1, 1)
        data.append([env.collapsed_qttts, policy, value])
        qttt, round_ctr, reward, done = env.act(i)
        if done:
            break
    network.train(data)
    policy, value = network.predict(env)
    print(policy, value)
