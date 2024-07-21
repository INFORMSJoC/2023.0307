from Trainer import Trainer
import torch
import random
import numpy as np
import sys
from Config import get_config
from concurrent import futures
from go.GoPlayers import *
from go.NNet import NNetWrapper as nn

def single_tread_sim(args, thread_ID):

    nnet = nn(args.boardsize, args)
    
    if(args.IterNumber != 1):
        nnet.load_checkpoint(args.load_nn[0], args.load_nn[1])

    t = Trainer(nnet, args.policy, args)
    t.SingleThreadSimulate(thread_ID)

if __name__=="__main__":
    
    IterNumber = int(sys.argv[1])
    
    seed = IterNumber
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.multiprocessing.set_start_method('spawn')
    
    parser = get_config(IterNumber)
    args = parser.parse_args(sys.argv[2:])

    futures_list = []
    
    with futures.ProcessPoolExecutor(max_workers=20) as executor:
        for i in range(5):
            a = executor.submit(single_tread_sim, args, i)
            futures_list.append(a)

        for item in futures.as_completed(futures_list):
            if item.exception() is not None:
                print(item.exception())
            else:
                print('success')
    