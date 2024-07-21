import math
import numpy as np
from copy import deepcopy as dc
import queue
import random
import matplotlib.pyplot as plt
import os
EPS = 1e-10

class Node():

    def __init__(self, game, from_action, father):
        self.game = game
        self.game_is_finished = self.game.getGameEnded()
        self.from_action = from_action
        self.father = father
        self.children = []
        self.N = 1
        self.V = None # NN prediction values for state after taking action a, V(game, a)
        self.Var = None # NN prediction variances for state after taking action a, V(game, a)
        self.Q = []   # Rollout value set, Q(father, from action)
        self.pi = None

    def set_child(self, child):
        self.children.append(child)
    
    def visit_time_add_one(self):
        self.N += 1

class MCTS():

    def __init__(self, nnet, policy, args):
        
        self.policy = policy
        self.nnet = nnet
        self.args = args
    
    def delete_tree(self, root_node):
        
        Q = queue.Queue()
        Q.put(root_node)
        all_nodes = []
        while(True):
            node = Q.get()
            all_nodes.append(node)
            for child in node.children:
                Q.put(child)
            if(Q.empty()):
                break
        for node in all_nodes:
            del node
        
    def getActionProb(self, game):
        
        root_node = Node(dc(game), from_action=None, father=None)

        for i in range(self.args.numMCTSSims):
            self.rollout(root_node)
        
        counts = [0 for i in range(game.getActionSize())]
        if self.args.policy == "UCT":
            for child in root_node.children:
                counts[child.from_action] = child.N
        else:
            valids = root_node.game.getValidMoves()
            if(root_node.pi is not None):
                pi = dc(root_node.pi)
            else:
                canonicalBoard = root_node.game.getBoardFeature(self.args.board_feature_channel)
                pi, v = self.nnet.predict(canonicalBoard)
                pi = pi*valids
                if(np.sum(pi) < EPS):
                    pi = 0*pi
                else:
                    pi /= np.sum(pi)
                root_node.pi = dc(pi)
            
            valids_one_hot = root_node.game.getValidMoves()
            valids = []
            for a in range(len(valids_one_hot)):
                if valids_one_hot[a]:
                    valids.append(a)

            if('Gaussian' in self.args.policy):
                ms = {}
                vas = {}
                ns = {}
                pv = {}
                pm = {}
                for a in valids:
                    for child in root_node.children:
                        if(child.from_action == a):
                            if(root_node.game.cur_player == 1):
                                ms[a] = np.mean(child.Q)
                            else:
                                ms[a] = 1 - np.mean(child.Q)
                            ns[a] = child.N
                            vas[a] = np.var(child.Q) + EPS
                            pv[a] = 1 / (1/self.args.sigmaa_0+ns[a]/vas[a])
                            pm[a] = pv[a]*(root_node.V[a]/self.args.sigmaa_0 + ns[a]*ms[a]/vas[a])
                            if('Pi' not in self.args.policy):
                                counts[a] = pm[a]
                            else:
                                counts[a] = pi[a]*pm[a]
            else:
                sm = {}
                pm = {}
                ns = {}
                alpha = {}
                beta = self.args.beta
                for a in valids:
                    for child in root_node.children:
                        if(child.from_action == a):
                            alpha[a] = beta/(1-root_node.V[a] + EPS)
                            if(root_node.game.cur_player == 1):
                                sm[a] = np.mean(child.Q)
                                pm[a] = (alpha[a] + child.N*sm[a])/(alpha[a] + beta + child.N)
                            else:
                                sm[a] = 1-np.mean(child.Q)
                                pm[a] = (alpha[a] + child.N*sm[a])/(alpha[a] + beta + child.N)
                            if('Pi' not in self.args.policy):
                                counts[a] = pm[a]
                            else:
                                counts[a] = pi[a]*pm[a]

        if self.args.StochasticAction == False or game.time_step >= self.args.exploreSteps:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            self.delete_tree(root_node)
            return probs
        
        counts = [x**0.5 for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        
        self.delete_tree(root_node)
        return probs
    
    def rollout(self, root_node):
        
        if(root_node.game_is_finished != -1):
            root_node.visit_time_add_one()
            return root_node.game_is_finished

        if(self.args.policy == 'UCT'):
            selected_act = self.UCT(root_node)
        elif('Gaussian' in self.args.policy):
            selected_act = self.AOAT_Gaussian(root_node)
        elif('Bernoulli' in self.args.policy):
            selected_act = self.AOAT_Bernoulli(root_node)
        
        is_leaf_node = True
        for child in root_node.children:
            if child.from_action == selected_act:
                is_leaf_node = False
                game_res = self.rollout(child)
                child.Q.append(game_res)
                root_node.visit_time_add_one()
                return game_res
        
        if is_leaf_node == True:
            
            new_game = dc(root_node.game)
            new_game.ExcuteAction(selected_act)
            new_node = Node(new_game, selected_act, root_node)
            root_node.set_child(new_node)
            root_node.visit_time_add_one()
            if(new_node.game_is_finished != -1):
                new_node.Q.append(new_node.game_is_finished)
                return new_node.game_is_finished
            else:
                canonicalBoard = new_game.getBoardFeature(self.args.board_feature_channel)
                _, v = self.nnet.predict(canonicalBoard)
                if(new_game.cur_player == 1):
                    actual_v = v
                else:
                    actual_v = 1-v
                new_node.Q.append(actual_v)
                return actual_v
    
    def get_node_next_state_Vs_Vars(self, valids, node):
        batch_states = []
        for a in valids:
            temp_game = dc(node.game)
            temp_game.ExcuteAction(a)
            new_board = temp_game.getBoardFeature(self.args.board_feature_channel)
            batch_states.append(new_board)
            del temp_game
        batch_Vs = self.nnet.predict(batch_states, is_batch = True)
        node.V = {}
        for i in range(len(valids)):
            node.V[valids[i]] = 1 - batch_Vs[i]

    def UCT(self, root_node):
        valids = root_node.game.getValidMoves()
        if(root_node.pi is not None):
            pi = dc(root_node.pi)
        else:
            canonicalBoard = root_node.game.getBoardFeature(self.args.board_feature_channel)
            pi, v = self.nnet.predict(canonicalBoard)
            pi = pi*valids
            if(np.sum(pi) < EPS):
                pi = 0*pi
            else:
                pi /= np.sum(pi)
            root_node.pi = dc(pi)
        
        all_valid_actions = []
        for a in range(len(valids)):
            if valids[a]:
                all_valid_actions.append(a)
        cur_best = -float('inf')
        best_act = -1
        for a in all_valid_actions:
            u = None
            for child in root_node.children:
                if(child.from_action == a):
                    if(root_node.game.cur_player == 1):
                        u = np.mean(child.Q) + self.args.cpuct*pi[a]*math.sqrt(root_node.N)/(1+child.N)
                    else:
                        u = 1-np.mean(child.Q) + self.args.cpuct*pi[a]*math.sqrt(root_node.N)/(1+child.N)
            if (u is None):
                u = 0.5 + self.args.cpuct*pi[a]*math.sqrt(root_node.N + EPS)
            if u > cur_best:
                cur_best = u
                best_act = a
        
        return best_act

    def AOAT_Gaussian(self, root_node):
        valids_one_hot = root_node.game.getValidMoves()
        valids = []
        for a in range(len(valids_one_hot)):
            if valids_one_hot[a]:
                valids.append(a)

        if(root_node.V is None):
            self.get_node_next_state_Vs_Vars(valids, root_node)
        
        if(len(valids) == 1):
            best_act = valids[0]
        else:
            ms = {}
            vas = {}
            ns = {}
            pv = {}
            pm = {}
            for a in valids:
                for child in root_node.children:
                    if(child.from_action == a):
                        if(root_node.game.cur_player == 1):
                            ms[a] = np.mean(child.Q)
                        else:
                            ms[a] = 1 - np.mean(child.Q)
                        ns[a] = child.N
                        vas[a] = np.var(child.Q) + EPS
                        pv[a] = 1 / (1/self.args.sigmaa_0+ns[a]/vas[a])
                        pm[a] = pv[a]*(root_node.V[a]/self.args.sigmaa_0 + ns[a]*ms[a]/vas[a])
                if(a not in ms):
                    ns[a] = 0
                    vas[a] = self.args.sigmaa_0
                    pv[a] = self.args.sigmaa_0
                    ms[a] = root_node.V[a]
                    pm[a] = root_node.V[a]

            A = max(pm, key=lambda x:pm[x])
            optimal_set = []
            for item in pm.items():
                if(item[1] == pm[A]):
                    optimal_set.append(item[0])
                    
            if(len(optimal_set) > 1):
                A = max(optimal_set, key=lambda n : vas[n] / (ns[n]+EPS)) 

            M = {}
            min_list = []
            for a in valids:
                if (a == A):
                    continue
                den = (pm[A] - pm[a])**2
                nv = 1 / ((root_node.V[A]*(1-root_node.V[A]) + EPS) + (ns[A]+1)/vas[A]) 
                num = nv + pv[a] 
                if (num < EPS):
                    min_list.append(float('inf'))
                else:
                    min_list.append(den/num)
            
            M[A] = np.min(min_list)

            for a in valids:
                if(a == A):
                    continue
                den = (pm[A] - pm[a])**2
                nv = 1 / (1/(root_node.V[a]*(1-root_node.V[a]) + EPS) + (ns[a]+1)/vas[a]) 
                num = pv[A] + nv
                if (num < EPS):
                    min_list = [float('inf')]
                else:
                    min_list = [den/num]
                for aa in valids:
                    if (aa == A or aa == a):
                        continue
                    den = (pm[aa] - pm[A])**2
                    num = pv[A] + pv[aa]
                    if (num < EPS):
                        min_list.append(float('inf'))
                    else:
                        min_list.append(den/num)
                M[a] = np.min(min_list)
                
            best_act = max(M, key=lambda x:M[x])
            
        return best_act
    
    def AOAT_Bernoulli(self, root_node):
        valids_one_hot = root_node.game.getValidMoves()
        valids = []
        for a in range(len(valids_one_hot)):
            if valids_one_hot[a]:
                valids.append(a)
                
        if(root_node.V is None):
            self.get_node_next_state_Vs_Vars(valids, root_node)
        
        if(len(valids) == 1):
            best_act = valids[0]
        else:
            
            sm = {}
            pm = {}
            ns = {}
            alpha = {}
            beta = self.args.beta
            for a in valids:
                for child in root_node.children:
                    if(child.from_action == a):
                        alpha[a] = beta/(1-root_node.V[a] + EPS)
                        if(root_node.game.cur_player == 1):
                            sm[a] = np.mean(child.Q)
                            pm[a] = (alpha[a] + child.N*sm[a])/(alpha[a] + beta + child.N)
                        else:
                            sm[a] = 1-np.mean(child.Q)
                            pm[a] = (alpha[a] + child.N*sm[a])/(alpha[a] + beta + child.N)
                        ns[a] = child.N
                if(a not in pm):
                    alpha[a] = beta/(1-root_node.V[a] + EPS)
                    pm[a] = root_node.V[a]
                    ns[a] = 0

            A = max(pm, key=lambda x:pm[x])
            optimal_set = []
            for item in pm.items():
                if(item[1] == pm[A]):
                    optimal_set.append(item[0])
                    
            if(len(optimal_set) > 1):
                A = min(optimal_set, key=lambda n : ns[n])
                
            M = {}
            min_list = []
            for a in valids:
                if (a == A):
                    continue
                den = (pm[a] - pm[A])**2
                num = (pm[A]*(1-pm[A]))/(ns[A]+2+alpha[A]+beta) + (pm[a]*(1-pm[a]))/(ns[a]+1+alpha[a]+beta) 
                if (num < EPS):
                    min_list.append(float('inf'))
                else:
                    min_list.append(den/num)
            
            M[A] = np.min(min_list)

            for a in valids:
                if(a == A):
                    continue
                den = (pm[a] - pm[A])**2
                num = (pm[A]*(1-pm[A]))/(ns[A]+1+alpha[A]+beta) + (pm[a]*(1-pm[a]))/(ns[a]+2+alpha[a]+beta)
                if (num < EPS):
                    min_list = [float('inf')]
                else:
                    min_list = [den/num]
                for aa in valids:
                    if (aa == A or aa==a):
                        continue
                    den = (pm[aa] - pm[A])**2
                    num = (pm[A]*(1-pm[A]))/(ns[A]+1+alpha[A]+beta) + (pm[aa]*(1-pm[aa]))/(ns[aa]+1+alpha[aa]+beta)
                    if (num < EPS):
                        min_list.append(float('inf'))
                    else:
                        min_list.append(den/num)
                        
                M[a] = np.min(min_list)
            best_act = max(M, key=lambda x:M[x])
        return best_act
    
      
        