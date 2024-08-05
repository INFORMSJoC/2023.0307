from go.GoGame import GoGame
from Arena import Arena
from MCTS import MCTS
import numpy as np
from utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import copy
import time

class Trainer():

    def __init__(self, nnet, policy, args):
        self.nnet = nnet
        self.pnet = self.nnet.__class__(args.boardsize, args)
        self.args = args
        self.policy = policy
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False
    
    def getPiWithNoise(self, pi):
        sample_num = 0
        for action_p in pi:
            if(action_p!=0):
                sample_num += 1
        noise = np.random.dirichlet([0.3] * sample_num)
        alpha = 0.25
        cnt = 0
        noisy_pi = []
        for action_p in pi:
            if(action_p==0):
                noisy_pi.append(action_p)
            else:
                noisy_pi.append(action_p*(1-alpha) + noise[cnt]*alpha)
                cnt += 1
        return noisy_pi
            
    def executeEpisode(self, mcts, game):

        trainExamples = []
        episodeStep = 0

        while True:
            episodeStep += 1            
            pi = mcts.getActionProb(game)            
            sym = game.getSymmetries(pi, self.args.board_feature_channel)
            if(game.time_step >= self.args.exploreSteps):
                for b, p in sym:
                    trainExamples.append([b, game.cur_player, p])
            noisy_pi = self.getPiWithNoise(pi)
            #for i in range(26):
            #    print(str(pi[i]) + ' ' + str(noisy_pi[i]))
            #print('---------------------------------------------------')
            action = np.random.choice(len(noisy_pi), p=noisy_pi)
            game.ExcuteAction(action)
            r = game.getGameEnded()
            if r!=-1:
                res_samples = []
                for x in trainExamples:
                    if(x[1] == 1):
                        res_samples.append((x[0],x[2],r))
                    else:
                        res_samples.append((x[0],x[2],1-r))
                return res_samples

    def LearnWithAllSamples(self):

        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)
        
        '''
        temp = []
        temp_5 = []
        cnt = 0
        for i in range(len(trainExamples)):
              if(trainExamples[i][2] != 0.5):
                  temp.append(trainExamples[i])
              else:
                  temp_5.append(trainExamples[i])
                  cnt += 1
        print(cnt/len(trainExamples))
        shuffle(temp_5)
        trainExamples = temp + temp_5[:int(0.2*len(temp_5))]
        '''
        
        shuffle(trainExamples)
        
        
        self.nnet.train(trainExamples, self.args.checkpoint + 'log.txt')
        
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, ID):
        return 'Tread_' + str(ID) + '_checkpoint' + '.pth.tar'

    def saveTrainExamples(self, ID):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(ID)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        for iter_samples_pth in self.args.samples_pths:
            all_examples = os.listdir(iter_samples_pth)
            for file_pth in all_examples:
                if file_pth[-8:] == 'examples':
                    examplesFile = self.args.load_folder_file[0] + file_pth
                    if not os.path.isfile(examplesFile):
                        print(examplesFile, "File with trainExamples not found.")
                    else:
                        print("File with trainExamples found. Read it.")
                        with open(examplesFile, "rb") as f:
                            self.trainExamplesHistory += Unpickler(f).load()
                        f.closed
    
    def SingleThreadSimulate(self, thread_ID):
        
        if not self.skipFirstSelfPlay:
            iterationTrainExamples = []

            eps_time = AverageMeter()
            bar = Bar('Self Play', max=self.args.numEps)
            end = time.time()

            for eps in range(self.args.numEps):

                mcts = MCTS(self.nnet, self.policy, self.args)
                game = GoGame(self.args.boardsize, self.args.game)

                iterationTrainExamples += self.executeEpisode(mcts, game)

                del mcts, game

                eps_time.update(time.time() - end)
                end = time.time()
                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                           total=bar.elapsed_td, eta=bar.eta_td)
                bar.next()
            bar.finish()

            self.trainExamplesHistory.append(iterationTrainExamples)

        self.saveTrainExamples(thread_ID)
        self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='nn.pth.tar')
