# import sys
# sys.path.append('E:\\CMU_INI\\11785\\project\\Qttt_RL')
from AlphaZero_Qttt.env_bridge import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset 
import numpy as np
from copy import deepcopy

from env import Env
import random
from rl_agent import TD_agent

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
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
            nn.Linear(512*3*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.policy_layer = nn.Linear(512, 72)
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
        output = torch.cat((X,Y), dim = 1)
        output = output.view(-1, 512*3*3)
        output = self.linear(output)
        policy = F.log_softmax(self.policy_layer(output), dim=1)
        value = torch.tanh(self.value_layer(output))

        return policy, value

class QtttDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        qttts, policy, value = self.data[index]
        if len(qttts) == 1:
            X, Y = qttts[0].to_tensor(), deepcopy(qttts[0]).to_tensor()
        else:
            X, Y = qttts[0].to_tensor(), qttts[1].to_tensor()
        return X, Y, torch.tensor(policy), torch.tensor(value)

class Network:
    def __init__(self, net):
        self.lr = 1e-3
        self.weight_decay = 5e-6
        self.epochs = 10
        self.batch_size = 512
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.net = net.to(self.device)
        

    def predict(self, game_env: EnvForDeepRL):
        """

        :param game_env: game env for evaluation
        :return:
            ndarray(72,) action_prob: all invalid action is masked to 0
            state_value(int): state_value of current env
        """
        self.net.eval()
        if len(game_env.collapsed_qttts) == 1:
            X, Y = game_env.collapsed_qttts[0].to_tensor(), deepcopy(game_env.collapsed_qttts[0]).to_tensor()
        else:
            X, Y = game_env.collapsed_qttts[0].to_tensor(), game_env.collapsed_qttts[1].to_tensor()

        X = X.unsqueeze(0).to(self.device)
        Y = Y.unsqueeze(0).to(self.device)

        output = self.net(X, Y)

        action_prob = output[0].detach().numpy()[0]# *game_env.valid_action_mask
        state_value = output[1].detach().numpy()[0]
        self.net.train()
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
        train_loader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=2)
        train_loader = DataLoader(train_dataset, **train_loader_args)

        criterion_p = nn.CrossEntropyLoss()
        criterion_v = nn.MSELoss()
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
                
                loss_p = criterion_p(policy, target_p)
                loss_v = criterion_v(value.view(-1), target_v)

                loss = loss_p + loss_v
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                del X
                del Y
                del target_p
                del target_v

            running_loss /= len(train_loader)
            print('epoch: ' , str(epoch + 1), 'Training Loss: ', running_loss)
        


if __name__ == '__main__':
    net = BasicNetwork()
    network = Network(net)
    env = Env()
    agent = TD_agent(1, 0, 1)
    data = []
    for i in range(5):
        policy = random.randint(0, 71)
        value = random.random() * ((-1)**random.randint(0,1))
        data.append([env.collapsed_qttts, policy, value])
        _, mark = env.get_state()
        free_qblock_id_lists, collapsed_qttts, _ = env.get_valid_moves()
        collapsed_qttt, agent_move = agent.act(free_qblock_id_lists, collapsed_qttts, mark)
        _, _, _, done = env.step(collapsed_qttt, agent_move, mark)
        if done:
            break
    network.train(data)
    policy, value = network.predict(env)
    print(policy, value)

        



