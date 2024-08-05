import argparse
import torch

def get_config(IterNumber):

    parser = argparse.ArgumentParser(description='AlphaGo Zero')

    parser.add_argument("--boardsize", type=int, default=3)
    parser.add_argument("--game", type=str, default='tic_tac_toe')
    parser.add_argument("--board_feature_channel", type=int,
                        default=5, choices=[5, 9])
    parser.add_argument("--numEps", type=int,
                        default=20)
    parser.add_argument("--updateThreshold", type=float,
                        default=0.5)
    parser.add_argument("--policy", type=str,
                        default='UCT', choices=['UCT', 'AOAT-Gaussian', 'AOAT-Bernoulli', 'AOAT-Bernoulli-Pi', 'AOAT-Bernoulli-Pi'])
    parser.add_argument("--StochasticAction", type=bool,
                        default=True, choices=[True, False])
    parser.add_argument("--numMCTSSims", type=int,
                        default=100)
    parser.add_argument("--cpuct", type=float,
                        default=1)
    parser.add_argument("--sigmaa_0", type=float,
                        default=0)
    parser.add_argument("--beta", type=int,
                        default=0)

    parser.add_argument("--checkpoint", type=str,
                        default='./temp/Iter' + str(IterNumber) + '/')
    parser.add_argument("--load_folder_file", type=tuple,
                        default=('./temp/Iter' + str(IterNumber) + '/','nn.pth.tar'))
    parser.add_argument("--load_nn", type=tuple,
                        default=('./temp/Iter' + str(IterNumber-1) + '/','best.pth.tar'))
    parser.add_argument("--IterNumber", type=int,
                        default=IterNumber)
    file_list = []
    
    
    if(IterNumber+1 <= 20):
        for i in range(1, IterNumber+1):
            file_list.append('./temp/Iter' + str(i) + '/')
    else:
        for i in range(IterNumber+1-20, IterNumber+1):
            file_list.append('./temp/Iter' + str(i) + '/')
    
    if(IterNumber+1 <= 15):
        exp = 2
    else:
        exp = 1
    parser.add_argument("--exploreSteps", type=int,
                        default=exp)
                        
    parser.add_argument("--samples_pths", type=int,
                        default=file_list)

    parser.add_argument("--lr", type=float,
                        default=0.0005)
    parser.add_argument("--dropout", type=float,
                        default=0.3)
    parser.add_argument("--epochs", type=int,
                        default=2)
    parser.add_argument("--batch_size", type=int,
                        default=64)
    parser.add_argument("--cuda", type=bool,
                        default=torch.cuda.is_available())
    parser.add_argument("--num_channels", type=int,
                        default=512)
                        
    return parser
