from Trainer import Trainer
import torch
import random
import numpy as np
import sys
from utils.utils import *
from Config import get_config
from go.GoGame import GoGame
from go.GoPlayers import *
from go.NNet import NNetWrapper as nn

if __name__=="__main__":
    
    IterNumber = int(sys.argv[1])
    
    seed = IterNumber
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    parser = get_config(IterNumber)
    args = parser.parse_args(sys.argv[2:])

    nnet = nn(args.boardsize, args)
    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    t = Trainer(nnet, args.policy, args)
    print("Load trainExamples from file")
    t.loadTrainExamples()
    t.LearnWithAllSamples()
    
