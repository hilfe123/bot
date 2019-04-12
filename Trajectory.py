import torch
class Trajectory:
    def __init__(self,tensor_sizes):
        self.mc_probs = torch.zeros([(tensor_sizes*30), 15])#.cuda()
        self.states = torch.zeros([(tensor_sizes*30), 49])#.cuda()
        self.states2 = torch.zeros([(tensor_sizes*30), 49])#.cuda()
        self.outcome_values = torch.zeros([(tensor_sizes*30),1])#.cuda()

    def insert(self,mc_prob,state,index):
        self.mc_probs[index] = mc_prob.squeeze()
        self.states[index] = torch.tensor(state).squeeze()#.cuda()

    def add_outcome_values(self,outcome_value,index):
        outcome_value = torch.tensor(outcome_value)#.cuda()
        self.outcome_values[index-29:index+1] = outcome_value