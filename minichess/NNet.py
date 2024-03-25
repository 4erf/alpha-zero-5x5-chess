import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

class MinichessNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(MinichessNNet, self).__init__()
        self.conv1 = nn.Conv2d(args.in_channels, 64, 3)
        self.conv2 = nn.Conv2d(64, 256, 3)
        # self.conv3 = nn.Conv2d(32, 32, 3)
        # self.conv4 = nn.Conv2d(32, 64, 3)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.bn4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(256, 128)
        self.fc_bn1 = nn.BatchNorm1d(128)

        # self.fc2 = nn.Linear(2048, 512)
        # self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(128, self.action_size)

        self.fc4 = nn.Linear(128, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, self.args.in_channels, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        # s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        # s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = s.view(-1, 256)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        # s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
