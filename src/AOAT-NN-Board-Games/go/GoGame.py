from __future__ import print_function
import sys
sys.path.append('..')
import numpy as np
import queue
import copy

class GoGame():
    def __init__(self, n, game):
        self.n = n
        if(game == 'tic_tac_toe'):
            self.win_num = 3
        elif(game == 'five_in_a_row'):
            self.win_num = 5
            
        self.directions = [(1,0),(0,-1),(-1,0),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]
        self.board = np.zeros([self.n, self.n])
        self.board_history = []
        self.time_step = 0

        self.cur_player = 1
    
    def copy_game(self):
        new_game = copy.deepcopy(self)
        return new_game

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.n*self.n
    
    def getBoardFeature(self, c):
        if(len(self.board_history) + 1 >= c):
            hiss = self.board_history[-(c-1):]
        else:
            hiss = self.board_history
        feat = []
        for his in hiss:
            feat.append(his['board'])
        feat.append(self.board)
        if(len(feat) < c):
            feat = [feat[0] for i in range(c-len(feat))] + feat
        player_indicate_layer = self.cur_player*np.ones([self.n, self.n])
        assembled_feat = [player_indicate_layer]
        for f in feat:
            assembled_feat.append(f)
        assembled_feat = np.stack(assembled_feat)
        return assembled_feat

    def ExcuteAction(self, action):
        self.time_step += 1
        if(action == self.n*self.n):
            self.board_history.append({'board': copy.deepcopy(self.board), 'player': self.cur_player})
            self.cur_player = -self.cur_player
        else:
            move = (int(action/self.n), action%self.n)
            x, y = move
            self.board_history.append({'board': copy.deepcopy(self.board), 'player': self.cur_player})
            self.board[x][y] = self.cur_player
            self.cur_player = -self.cur_player 

    def getValidMoves(self):

        moves = []
        for y in range(self.n):
            for x in range(self.n):
                if self.board[x][y] == 0:
                    moves.append(self.n*x + y)   

        valids = [0]*self.getActionSize()
        for move in moves:
            valids[move] = 1
        return np.array(valids)
    
    def _in_board(self, x, y):
        if(x<0 or x>=self.n or y<0 or y >= self.n):
            return False
        else:
            return True
        
    def _judge_win(self, color, piece):
        for direction in self.directions:
            line = [(piece[0] + i*direction[0], piece[1] + i*direction[1]) for i in range(self.win_num)]
            flag = 0
            for p in line:
                if self._in_board(p[0], p[1]) == False or self.board[p[0]][p[1]] != color:
                    flag = 1
                    break
            if(flag == 0):
                return True
        return False

    def _find_winner(self):
        winner = 0
        for x in range(self.n):
            for y in range(self.n):
                if(self.board[x][y] != 0):
                    if(self._judge_win(self.board[x][y], (x, y))):
                        winner = self.board[x][y]
                        return winner
        return winner

    def getGameEnded(self):

        valid_moves = self.getValidMoves()
        if(np.sum(valid_moves) == 0):
            return 0.5
        else:
            winner = self._find_winner()
            if(winner == 1):
                return 1
            elif(winner == -1):
                return 0
            else:
                return -1

    def getSymmetries(self, pi, c):
        pi_board = np.reshape(pi, (self.n, self.n))
        l = []
        board_feat = self.getBoardFeature(c)
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board_feat, i, axes=(1,2))
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.transpose(np.fliplr(np.transpose(newB, (0, 2, 1))), (0, 2, 1))
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()))]
        return l

    def display(self):
        n = self.n

        for y in range(n):
            print (y,"|",end="")
        print("")
        print(" -----------------------")
        for y in range(n):
            print(y, "|",end="")    # print the row #
            for x in range(n):
                piece = self.board[y][x]    # get the piece to print
                if piece == -1: print("w ",end="")
                elif piece == 1: print("b ",end="")
                else:
                    if x==n:
                        print("-",end="")
                    else:
                        print("- ",end="")
            print("|")

        print("   -----------------------")




