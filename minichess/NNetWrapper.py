import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim

import chess

from .NNet import MinichessNNet

piece_channel = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.QUEEN: 2,
    chess.KING: 3,
}

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'weight_decay': 0.001,
    'cuda': torch.cuda.is_available(),
    'in_channels': len(piece_channel)
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = MinichessNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        print(f'Params: {sum(p.numel() for p in self.nnet.parameters() if p.requires_grad)}')

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), weight_decay=args['weight_decay'])

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.stack([self.board_to_tensor(board) for board in boards])
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board: chess.Board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = self.board_to_tensor(board)
        if args.cuda: board = board.contiguous().cuda()
        # board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

    def clamp(self, value):
        # Shouldn't really be used but to be safe
        return max(min(value, self.board_x - 1), 0)

    def board_to_tensor(self, board: chess.Board):
        tensors = [torch.FloatTensor() for _ in piece_channel.keys()]

        for piece, channel in piece_channel.items():
            tensor = [[0.0 for _ in range(0, self.board_x)] for _ in range(0, self.board_y)]

            for s in board.pieces(piece, chess.WHITE):
                x = self.clamp(chess.square_file(s))
                y = self.clamp(chess.square_rank(s))
                tensor[y][x] = 1.0

            for s in board.pieces(piece, chess.BLACK):
                x = self.clamp(chess.square_file(s))
                y = self.clamp(chess.square_rank(s))
                tensor[y][x] = -1.0

            tensors[channel] = torch.FloatTensor(np.array(tensor, dtype=np.float64))

        return torch.stack(tensors)