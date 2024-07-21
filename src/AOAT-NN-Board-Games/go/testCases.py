from GoGame import GoGame
import random

def random_sim(seed):
    random.seed(seed)
    game = GoGame(5)
    cnt = 0
    while True:
        cnt += 1
        if(cnt > 200):
            return 1
        validmoves = game.getValidMoves()
        chosable = []
        game.display()
        print(game.cur_player)
        print(game.getValidMoves())
        a = int(input("Action: "))
        game.ExcuteAction(a)
        e = game.getGameEnded()
        if(e != -1):
            return 0

random_sim(18)



