import Arena
from MCTS import MCTS
from go.GoGame import GoGame
from go.GoPlayers import *
from go.NNet import NNetWrapper as NNet
import numpy as np
import torch
import random
from Config import get_config
import numpy as np
from utils.utils import *
import sys
from concurrent import futures

def search(thread_id, search_budgets, UCT_params, AOAP_params, iter_models, policy):
    for search_budget in search_budgets:
        for UCT_param in UCT_params:
            for aoap_param in AOAP_params:
                for model_index in iter_models:
                    parser1 = get_config(1)
                    args1 = parser1.parse_args(sys.argv[2:])
                    if('Gaussian' in policy):
                        args1.sigmaa_0 = aoap_param
                    else:
                        args1.beta = aoap_param
                    args1.policy = policy
                    args1.StochasticAction = True
                    n1 = NNet(args1.boardsize, args1)
                    n1.load_checkpoint('./temp/Iter'+str(model_index)+'/','best.pth.tar')
                    args1.numMCTSSims = search_budget
                    mcts1 = MCTS(n1, args1.policy, args1)
                    n1p = lambda x: mcts1.getActionProb(x)
                    
                    parser2 = get_config(1)
                    args2 = parser2.parse_args(sys.argv[2:])
                    args2.cpuct = UCT_param
                    args2.policy = "UCT"
                    n2 = NNet(args2.boardsize, args2)
                    args2.StochasticAction = True
                    n2.load_checkpoint('./temp/Iter'+str(model_index)+'/','best.pth.tar')
                    args2.numMCTSSims = search_budget
                    mcts2 = MCTS(n2, args2.policy, args2)
                    n2p = lambda x: mcts2.getActionProb(x)
                    
                    arena = Arena.Arena(n2p, n1p, args1)
                    one_win, two_win, diffs = arena.playGames(4, verbose=False)
                    wins[(search_budget, UCT_param, aoap_param, model_index)] = [one_win, two_win]
                    with open('./pit_' + str(thread_id) + '.txt', 'a+') as f:
                        f.write(str(search_budget) + ' ' + str(UCT_param) + ' ' + str(aoap_param) + ' ' + str(model_index) + ' ' + str(one_win) + ' ' + str(two_win) + ' ' + str(diffs[0]) + ' ' + str(diffs[1]) + '\n')


seed = int(sys.argv[1])
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

search_budgets = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] # evaluation search budgets
UCT_params = [0.2*(i+1) for i in range(5)] # param for UCT
AOAP_params = [0.05*(i+1) for i in range(5)] # param for AOAT-Gaussian
#AOAP_params = [1, 2, 3, 4, 5, 6] # param for AOAT-Bernoulli
iter_models = [[31]] # indexes for evaluation models
policy = 'AOAT-Gaussian' # target policy to compete with UCT; 'AOAT-Gaussian' or 'AOAT-Bernoulli' or 'AOAT-Bernoulli-Pi' or 'AOAT-Bernoulli-Pi'

wins = {}
cnt = 0
futures_list = []
with futures.ProcessPoolExecutor(max_workers=20) as executor:
    for iter_model in iter_models:
        a = executor.submit(search, cnt, search_budgets, UCT_params, AOAP_params, iter_model, policy)
        futures_list.append(a)
        cnt += 1
    
    for item in futures.as_completed(futures_list):
        if item.exception() is not None:
            print(item.exception())
        else:
            print('success')
