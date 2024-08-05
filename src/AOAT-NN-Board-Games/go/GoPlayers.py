import numpy as np
import random

class HumanGoPlayer():

    def play(self, game):
        game.display()
        a = int(input('Input action: '))
        a_prob = np.zeros(game.getActionSize())
        a_prob[a] = 1
        return a_prob

class RandomPlayer():

    def play(self, game):
        valids_one_hot = game.getValidMoves()
        valids = []
        for a in range(len(valids_one_hot)):
            if valids_one_hot[a]:
                valids.append(a)
        random.shuffle(valids)
        a_prob = [0 for i in range(len(valids_one_hot))]
        a_prob[valids[0]] = 1
        return a_prob