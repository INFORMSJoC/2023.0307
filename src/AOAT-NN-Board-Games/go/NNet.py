import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../../')
from utils.utils import *
from utils import Bar, AverageMeter

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .GoNNet import GoNNet as onnet

class NNetWrapper():
    def __init__(self, n, args):
        self.nnet = onnet(n, args)
        self.board_x = n
        self.board_y = n
        self.action_size = n*n+1
        self.args = args

        if self.args.cuda:
            self.nnet.cuda()
            
    def train(self, examples, log_path):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr = self.args.lr, weight_decay = 1e-3)
        #print(optimizer.state_dict()['param_groups'][0]['lr'])
        
        with open(log_path, 'w+') as f:
            log_string = 'batch size data bt total eta lpi lv \n'
            f.write(log_string)
            for epoch in range(self.args.epochs):
                print('EPOCH ::: ' + str(epoch+1))
                self.nnet.train()
                data_time = AverageMeter()
                batch_time = AverageMeter()
                pi_losses = AverageMeter()
                v_losses = AverageMeter()
                end = time.time()
    
                bar = Bar('Training Net', max=int(len(examples)/self.args.batch_size))
                batch_idx = 0
    
                while batch_idx < int(len(examples)/self.args.batch_size):
                    sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                    boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                    boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                    target_pis = torch.FloatTensor(np.array(pis))
                    target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
    
                    # predict
                    if self.args.cuda:
                        boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                    boards, target_pis, target_vs = Variable(boards), Variable(target_pis), Variable(target_vs)
    
                    # measure data loading time
                    data_time.update(time.time() - end)

                    # compute output
                    out_pi, out_v = self.nnet(boards)
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)
                    total_loss = l_pi + l_v
    
                    # record loss
                    pi_losses.update(l_pi.item(), boards.size(0))
                    v_losses.update(l_v.item(), boards.size(0))
    
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
    
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    batch_idx += 1
    
                    # plot progress
                    log_string = '{batch} {size} {data:.3f} {bt:.3f} {total:} {eta:} {lpi:.4f} {lv:.3f}\n'.format(
                                batch=batch_idx,
                                size=int(len(examples)/self.args.batch_size),
                                data=data_time.avg,
                                bt=batch_time.avg,
                                total=bar.elapsed_td,
                                eta=bar.eta_td,
                                lpi=pi_losses.avg,
                                lv=v_losses.avg,
                                )
                    f.write(log_string)
                    bar.next()
                bar.finish()

    def predict(self, board, is_batch = False):

        start = time.time()

        # preparing input
        if(is_batch == False):
            board = torch.FloatTensor(board.astype(np.float64))
            if self.args.cuda: board = board.contiguous().cuda()
            with torch.no_grad():
                board = Variable(board)
            board = board.view(1, board.shape[0], board.shape[1], board.shape[2])
        else:
            batch_board = []
            for b in board:
                bb = torch.FloatTensor(b.astype(np.float64))
                if self.args.cuda: bb = bb.contiguous().cuda()
                with torch.no_grad():
                    bb = Variable(bb)
                bb = bb.view(1, bb.shape[0], bb.shape[1], bb.shape[2])
                batch_board.append(bb)
            board = torch.cat(batch_board, dim = 0)

        self.nnet.eval()
        
        pi, v = self.nnet(board)
        if(is_batch == False):
            return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]
        else:
            v = v.data.cpu().numpy()
            v = [v[i][0] for i in range(v.shape[0])]
            return v

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath)
        self.nnet.load_state_dict(checkpoint['state_dict'])
