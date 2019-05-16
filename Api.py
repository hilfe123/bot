import torch
from Network import Network
from Environment import Environment
from Tree import Tree as Tree
from Node import Node
import copy
import numpy as np
from operator import attrgetter

class Api:
    def __init__(self):
        self.network = Network(49,128)
        self.network.load_state_dict(torch.load(r'C:\Users\Abgedrehter Alman\PycharmProjects\bot\models and trajectory\model3'))
        self.network.eval()
        self.env = Environment()
        self.tree = Tree(self.network)
        self.tree.init_tree()

    def build_states(self,state,bot_action,enemy_action,env_action,black_score,white_score):
        new_black_state =  copy.copy(state)
        new_black_state[-1]=white_score
        new_black_state[-2]=black_score
        new_black_state[-3]=0
        new_black_state[env_action+35]=0
        new_black_state[bot_action-1] = 0
        new_black_state[enemy_action+14] = 0

        new_white_state = copy.copy(new_black_state)
        new_white_state[-3]=1
        return new_black_state,new_white_state

    def set_probabilities(self, node):
        tensor = torch.from_numpy(np.array(node.state)).unsqueeze(dim=0).float()
        prediction_value, probabilities = self.network(tensor)
        probabilities = probabilities.detach().t()
        prediction_value = prediction_value.detach()
        node.probabilities = [ probabilities[action-1].item() for action in node.actionspace]
        node.v = prediction_value

    def do_action(self,random_action,env_action):
        for i in range(0,1600):
            self.tree.env.set_env(self.tree.black_root.state)
            self.tree.double_agent_simulation()
        action_edge = max(self.tree.black_root.edges,key=attrgetter('visits'))
        bot_action = action_edge.action
        bot_score,random_score = self.env.update_values(bot_action,random_action,env_action)
        self.tree.env.update_values(bot_action,random_action,env_action)

        black_state , white_state = self.build_states(self.tree.black_root.state,bot_action,random_action,env_action,bot_score,random_score)
        black_root = Node(black_state)
        black_root.build_actions()
        self.set_probabilities(black_root)
        print(black_root.v.item())

        white_root = Node(white_state)
        white_root.build_actions()
        self.set_probabilities(white_root)

        self.tree.black_root = black_root
        self.tree.white_root = white_root

        done = self.env.check_status()
        if done:
            self.tree.init_tree()
            self.env.reset_env()
        return done,bot_action,bot_score,random_score


