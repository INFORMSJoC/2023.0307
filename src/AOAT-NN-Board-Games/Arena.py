import numpy as np
from utils import Bar, AverageMeter
from go.GoGame import GoGame
import time

class Arena():
    
    def __init__(self, player1, player2, args = None):

        self.player1 = player1
        self.player2 = player2
        self.args = args

    def playGame(self, verbose = False):
        
        players = [self.player2, None, self.player1]
        game = GoGame(self.args.boardsize, self.args.game)
        it = 0
        print(it)
        while True:
            it+=1
            if verbose:
                print("Turn ", str(it), "Player ", str(game.cur_player))
                game.display()
            pi = players[game.cur_player+1](game)
            action = np.random.choice(len(pi), p=pi)
            game.ExcuteAction(action)
            e = game.getGameEnded()
            if(e != -1):
                break
        #diff = game.countDiff()
        diff = 0
        del game
        return e, diff

    def playGames(self, num, verbose=False):
        
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        diffs = []
        
        for _ in range(num):
            gameResult, diff = self.playGame(verbose=verbose)
            if gameResult==1:
                oneWon+=1
            elif gameResult==0:
                twoWon+=1
            diffs.append(diff)
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1
        
        for _ in range(num):
            gameResult, diff = self.playGame(verbose=verbose)
            if gameResult==0:
                oneWon+=1                
            elif gameResult==1:
                twoWon+=1
            diffs.append(diff)
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
            
        bar.finish()

        return oneWon, twoWon, diffs
