import numpy
from abc import ABC, abstractmethod
from collections import defaultdict
from numpy import  log, sqrt

class MCTS:
    
    def __init__(self, exploration_weight=1, policy='uct', budget=1000, n0=4, opp_policy='random', muu_0=2, sigmaa_0=2, sigma_0=1):
        self.Q = defaultdict(float) 
        self.V_bar = defaultdict(float) 
        self.V_hat = defaultdict(float)
        self.N = defaultdict(int) 
        self.children = defaultdict(set)  
        self.exploration_weight = exploration_weight 
        self.all_Q = defaultdict(list)  
        assert policy in {'uct', 'ocba','AOAP'}
        self.policy = policy
        self.std = defaultdict(float) 
        self.ave_Q = defaultdict(float) 
        self.pv = defaultdict(float)
        self.pm = defaultdict(float)
        self.budget=budget 
        self.n0 = n0
        self.leaf_cnt = defaultdict(int) 
        self.opp_policy = opp_policy
        self.muu_0 = muu_0
        self.sigmaa_0 = sigmaa_0
        self.sigma_0 = sigma_0
        

    def choose(self, node):
        if node.is_terminal():
            raise RuntimeError("choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child() 

        def score(n):
            if self.N[n] == 0: 
                return float("-inf")  
            if self.policy == 'uct':
                return self.ave_Q[n] 
            elif self.policy == 'AOAP':
                return self.pm[n] 
            else:
                return self.ave_Q[n] 
        rtn = max(self.children[node], key=score) 
        
        return rtn

    def do_rollout(self, node):
        
        path = self._select(node)
        leaf = path[-1]
        
        self.leaf_cnt[leaf] += 1 
        sim_reward = self._simulate(leaf)
        self._backpropagate(path, sim_reward)

    def _select(self, node):
        
        path = []
        while True:
            path.append(node) 
            self.N[node] += 1 
            if node.terminal:
                return path
            
            if len(self.children[node]) == 0:
                children_found = node.find_children()
                self.children[node] = children_found
                
            
            if node.turn == -1:
                if self.opp_policy == 'random':
                    node = node.find_random_child()
                if self.opp_policy == 'uct':  
                    expandable = [n for n in self.children[node] if self.N[n] < 1] 
                    if expandable: 
                        node = expandable.pop() 
                    else:
                        log_N_vertex = log(sum([self.N[c] for c in self.children[node]])) 
                        node = min(self.children[node], key=lambda n:self.ave_Q[n] 
                                   - self.exploration_weight * sqrt( 2 * log_N_vertex / self.N[n]))
                continue
            
            expandable = [n for n in self.children[node] if self.N[n] < self.n0] 
            if  expandable:
                a = self._expand(node)
                if len(self.children[a]) == 0:
                    self.children[a] = a.find_children()
                path.append(a)
                self.N[a] += 1 
                
                return path
            else: 
                if self.policy == 'uct':
                    a = self._uct_select(node) 
                elif self.policy == 'AOAP':
                    a = self._AOAP_select(node)
                else:
                    a = self._ocba_select(node)
                node = a

    def _expand(self, node, path_reward=None):
        explored_once = [n for n in self.children[node] if self.N[n] < self.n0]
        return explored_once.pop()

    def _simulate(self, node):
        while True:
            if not node.is_terminal(): 
                node = node.find_random_child()
            if node.terminal: 
                return node.reward()

    def _backpropagate(self, path, r):
        for i in range(len(path)-1, -1, -1):
            node = path[i] 
            self.Q[node] += r 
            self.all_Q[node].append(r)
            
            old_ave_Q = self.ave_Q[node] 
            self.ave_Q[node] = self.Q[node] / self.N[node] 
            self.std[node] = sqrt(((self.N[node]-1)*self.std[node]**2 + (r - old_ave_Q) * (r - self.ave_Q[node]))/self.N[node]) 
            
            if self.std[node] == 0:
               self.std[node] = self.sigma_0
               
            self.pv[node] = 1 / (1/self.sigmaa_0+self.N[node]/(self.std[node])**2)  
            self.pm[node] = self.pv[node]*(self.muu_0/self.sigmaa_0+self.N[node]*self.ave_Q[node]/(self.std[node])**2)

    def _uct_select(self, node):
        
        assert all(n in self.children for n in self.children[node]) 

        log_N_vertex = log(sum([self.N[c] for c in self.children[node]])) 

        def uct(n):
            return self.ave_Q[n] + self.exploration_weight * sqrt(
                2 * log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
    
    def _ocba_select(self, node):
        assert all(n in self.children for n in self.children[node])
        assert len(self.children[node])>0
        
        if len(self.children[node]) == 1:
            return list(self.children[node])[0]
        
        all_actions = self.children[node] 
        b = max(all_actions, key=lambda n: self.ave_Q[n]) 
        best_Q = self.ave_Q[b] 
        suboptimals_set, best_actions_set, select_actions_set = set(), set(), set() 
        for k in all_actions:
            if self.ave_Q[k] == best_Q:
                best_actions_set.add(k) 
            else:
                suboptimals_set.add(k)
        
        if len(suboptimals_set) == 0:
            return min(self.children[node], key=lambda n: self.N[n]) 
        
        if len(best_actions_set) != 1:
            b = max(best_actions_set, key=lambda n : (self.std[node])**2 / self.N[n]) 
            
        for k in all_actions:
            if self.ave_Q[k] != best_Q:
                select_actions_set.add(k)
        select_actions_set.add(b)
        
        delta = defaultdict(float) 
        for k in select_actions_set:
            delta[k] = self.ave_Q[k] - best_Q 
        
        ref = next(iter(suboptimals_set)) 

        para = defaultdict(float)
        ref_std_delta = self.std[ref]/delta[ref] 
        para_sum = 0
        for k in suboptimals_set:
            para[k] = ((self.std[k]/delta[k])/(ref_std_delta))**2 
               
        para[b] = sqrt(sum((self.std[b]*para[c]/self.std[c])**2 for c in suboptimals_set))

        para_sum = sum(para.values()) 
        para[ref] = 1
       
        totalBudget = sum([self.N[c] for c in select_actions_set])+1
        ref_sol = (totalBudget)/para_sum
        
        return max(select_actions_set, key=lambda n:para[n]*ref_sol - self.N[n])

    def _AOAP_select(self, node):
        
        assert all(n in self.children for n in self.children[node])
        assert len(self.children[node]) > 0
        
        if len(self.children[node]) == 1:
            return list(self.children[node])[0] 
        
        all_actions = self.children[node]  
        b = max(all_actions, key=lambda n: self.pm[n])  
        best_Q = self.pm[b]  
        suboptimals_set, best_actions_set, select_actions_set = set(), set(), set()  
        
        for k in all_actions:
            if self.pm[k] == best_Q:
                best_actions_set.add(k)           
            else:
                suboptimals_set.add(k)
                
        if len(suboptimals_set) == 0:
            return min(self.children[node], key=lambda n: self.N[n]) 
        
        if len(best_actions_set) != 1:
            b = max(best_actions_set, key=lambda n : self.pv[n] / self.N[n]) 
            
        for k in all_actions:
            if self.pm[k] != best_Q:
                select_actions_set.add(k)
        select_actions_set.add(b)
            
        M = defaultdict(int)
        V = defaultdict(float)
        W = defaultdict(float)
        nv = defaultdict(float)
        
        
        for k in all_actions:
            nv[k] = self.pv[k]
            M[k] = self.N[k]
        
        for k in select_actions_set:
            M[k] += 1
            nv[k] = 1 / (1/self.sigmaa_0 + M[k]/(self.std[k])**2)  
            for i in suboptimals_set:
                W[i] = (best_Q - self.pm[i])**2 / (nv[b] + nv[i])
            V[k] = min(W.values()) 
            M[k] -= 1
            nv[k] = self.pv[k]
                  
        return max(select_actions_set, key=lambda n:V[n])
    
class Node(ABC):

    @abstractmethod
    def find_children(self):
        return set()

    @abstractmethod
    def find_random_child(self):
        return None

    @abstractmethod
    def is_terminal(self):
        return True

    @abstractmethod
    def reward(self):
        return 0

    @abstractmethod
    def __hash__(self):
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        return True