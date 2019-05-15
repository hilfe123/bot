import torch
class Trajectory:
    def __init__(self,tensor_sizes):
        self.mc_probs = torch.zeros([(tensor_sizes*30), 15])
        self.states = torch.zeros([(tensor_sizes*30), 49])
        self.outcome_values = torch.zeros([(tensor_sizes*30),1])

    def insert(self,mc_prob,state,index):
        self.mc_probs[index] = torch.tensor(mc_prob).squeeze()
        self.states[index] = torch.tensor(state).squeeze()

    def add_outcome_values(self,outcome_value,index):
        outcome_value = torch.tensor(outcome_value)
        self.outcome_values[index-30:index] = outcome_value