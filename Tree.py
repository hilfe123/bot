import numpy as np
from Node import Node
from Environment import Environment
import random
import torch
import copy
import math

class Tree:
    def __init__(self, network):
        self.network = network
        self.root = 0
        self.env = Environment()


    def init_tree(self):
        self.root = Node(
            np.array([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0])
        )
        self.set_probabilities(self.root)
        self.root.build_actionspace()
        self.root.build_edges()

    def update_values(self, v, edges):
        for edge in edges:
            edge.visits += 1
            edge.totalactionvalue += v
            edge.actionvalue = edge.totalactionvalue / edge.visits

    def set_probabilities(self, node):
        mirror_state = copy.copy(node.state)
        if node.state[-3] == 1:
            mirror_state[-3] = 0
        else: mirror_state[-3] = 1
        tensor = torch.zeros(2,49)
        tensor[0] = torch.from_numpy(np.array(node.state)).float()
        tensor[1] = torch.from_numpy(np.array(mirror_state)).float()
        prediction_value, probabilities = self.network(tensor)
        probabilities = probabilities.detach()
        prediction_value = prediction_value.detach()
        node.probabilities = probabilities[0]
        node.mirror_probs = probabilities[1]
        node.v = prediction_value[0]

    """
    Sample actions from the probability distribution of the mirror state [1] instead of [0] or otherwise.
    Actionspace of alternative player is contained in node state and the mirror probs are saved when the "normal"
    probability distribution was derived from the network.   
    """

    def derive_action(self,black_player,node):

        if black_player:
            local_actionspace = node.state[15:30]
        else:
            local_actionspace = node.state[0:15]

        local_actionspace = [value for value in local_actionspace if value != 0]

        probs = [node.mirror_probs[i - 1] for i in local_actionspace]
        probs = [item.item() for item in probs]
        probs = np.array(probs)
        probs /= probs.sum()

        action = np.random.choice(local_actionspace , p=probs)
        return action

    def build_node(self,state,env_action,black_action,white_action,black_score,white_score):
        new_state = copy.copy(state)
        new_state[black_action - 1] = 0
        new_state[white_action + 14] = 0
        new_state[env_action + 35] = 0
        new_state[-2] = black_score
        new_state[-1] = white_score
        node = Node(new_state)
        node.build_actionspace()
        node.build_edges()
        return node


    def simulation(self):

        self.env.reset49(self.root.state[-2],self.root.state[-1])
        self.env.update_env(self.root.state[30:46])
        self.env.reset_deck()
        child_added = False
        node = self.root
        done = False
        edge_buffer = []
        c = 0.75

        while not done:

            black_player = True if node.state[-3]==0 else False

            actionvalues = [edge.actionvalue for edge in node.edges]
            visits = [edge.visits for edge in node.edges]
            total_visits = sum(visits)
            total_visits_sqrt = math.sqrt(total_visits)

            actions = node.actionspace
            probs = [node.probabilities[i - 1] for i in actions]

            upper_bounds = [(c * prob.item() * (total_visits_sqrt / (1 + visit))) for prob, visit in zip(probs, visits)]

            if node.state[-3] == 1:
                upper_bounds = [-bound for bound in upper_bounds]

            edge_values = [q + u for q, u in zip(actionvalues, upper_bounds)]

            if all(x == edge_values[0] for x in edge_values):
                chosen_edge = random.choice(node.edges)

                if black_player:
                    black_action = chosen_edge.action
                    white_action = self.derive_action(black_player,node)
                else:
                    white_action = chosen_edge.action
                    black_action = self.derive_action(black_player,node)

            else:

                if black_player:
                    action_index = edge_values.index(max(edge_values))
                    chosen_edge = node.edges[action_index]
                    black_action = chosen_edge.action
                    white_action = self.derive_action(black_player,node)
                else:
                    action_index = edge_values.index(min(edge_values))
                    chosen_edge = node.edges[action_index]
                    white_action = chosen_edge.action
                    black_action = self.derive_action(black_player,node)

            env_action = self.env.step(black_action,white_action)
            black_score,white_score = self.env.get_player_scores()
            new_node = self.build_node(node.state,env_action,black_action,white_action,black_score,white_score)
            contains = False

            for loop_node in chosen_edge.targetNodes:
                #[0:30] , but node = new_node to simulate further
                #also need to change build_node but otherwise parallel to this class
                if all(loop_node.state == new_node.state):
                    edge_buffer.append(chosen_edge)
                    node = loop_node
                    contains = True
                    break

            if not contains and not child_added:
                self.set_probabilities(new_node)
                chosen_edge.targetNodes.append(new_node)
                edge_buffer.append(chosen_edge)
                node = new_node
                break

            done = self.env.check_status()
        self.update_values(node.v, edge_buffer)






