import torch
from Network import Network
from Environment import Environment
from Tree import Tree
from Node import Node
import copy
import numpy as np
from operator import attrgetter

class Api:
    def __init__(self):
        self.network = Network(49,256)
        self.env = Environment()
        self.tree = Tree(self.network)
        self.tree.init_tree()

    def init_net(self):
        self.network.load_state_dict(torch.load(r'C:\Users\Abgedrehter Alman\PycharmProjects\AGZ\model3'))

    def build_state(self,node,action0,action1,env_action,player0score,player1score):
        new_state = copy.copy(node.state)
        new_state[action0 - 1] = 0
        new_state[action1 + 14] = 0
        new_state[env_action + 35] = 0
        new_state[-1] = player1score
        new_state[-2] = player0score
        return new_state

    def do_action(self,random_action,env_action):
        new_root = 0
        node = self.tree.root
        for i in range(0,800):
            self.tree.simulation()
        action_edge = max(node.edges,key=attrgetter('visits'))
        bot_action = action_edge.action
        bot_score,random_score = self.env.update_values(bot_action,random_action,env_action)
        x,y = self.tree.env.update_values(bot_action,random_action,env_action)
        new_state = self.build_state(self.tree.root,bot_action,random_action,env_action,bot_score,random_score)

        for loop_node in action_edge.targetNodes:
            if all(loop_node.state == new_state):

                new_root = loop_node
                break

        if new_root == 0:

            new_root = Node(new_state)
            new_root.build_actionspace()
            new_root.build_edges()

            tensor = torch.zeros(2,49)
            mirror_state = copy.copy(new_state)
            mirror_state[-3] = 1
            tensor[0] = torch.from_numpy(np.array(new_state)).float()
            tensor[1] = torch.from_numpy(np.array(mirror_state)).float()
            prediction_value, probabilities = self.network(tensor)
            probabilities = probabilities.detach()
            new_root.probabilities = probabilities[0]
            new_root.mirror_probs = probabilities[1]

        self.tree.root = new_root



        done = self.env.check_status()
        if done:
            self.tree.init_tree()
            self.env.reset_env()
        return done,bot_action,bot_score,random_score


