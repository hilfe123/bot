import numpy as np
from Node import Node
from Environment import Environment
import random
import torch
import copy
import math


class Tree:
    def __init__(self,network):
        self.network = network
        self.root = 0
        self.env = Environment()
        self.mirror_root = 0

    def setNetwork(self,network):
        self.network = network

    def init_tree(self):
        self.root = Node(
            np.array([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0])
        )
        self.set_probabilities(self.root)
        self.root.build_actions()

    def update_values(self, v,edge_buffer):
        for edge in edge_buffer:
            edge.visits += 1
            edge.totalactionvalue += v
            edge.actionvalue = edge.totalactionvalue / edge.visits

    def update_values_double_root(self,nodes,edge_buffer):
        v = [node.v for node in nodes]

        for edge in edge_buffer:

            if edge_buffer.index(edge)%2 == 0:
                index = 0
            else: index =1

            edge.visits +=1
            edge.totalactionvalue += v[index].item()
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
        #noise = [1-prob for prob in probs ]
        #noise = np.array(noise)
        #noise /= noise.sum()

        #final_probs = [0.2*probs + 0.8*noise for probs,noise in zip(probs,noise)]
        #final_probs = np.array(final_probs)
        #final_probs /= final_probs.sum()
        #if api:
        #probs = [1/probs.size for x in probs]
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
        node.build_actions()
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
            edge_buffer.append(chosen_edge)

            for loop_node in chosen_edge.targetNodes:
                if all(loop_node.state == new_node.state):
                    node = loop_node
                    contains = True
                    break

            if not contains and not child_added:
                self.set_probabilities(new_node)
                chosen_edge.targetNodes.append(new_node)
                node = new_node
                break

            done = self.env.check_status()
        self.update_values(node.v.item(), edge_buffer)

    def double_root_simulation(self):

        self.env.reset49(self.root.state[-2],self.root.state[-1])
        self.env.update_env(self.root.state[30:46])
        self.env.reset_deck()
        child_added = False
        done = False
        edge_buffer = []
        edge_index = 0
        c = 0.8
        epsilon = 0.25

        if self.mirror_root == 0:
            mirror_state = copy.copy(self.root.state)
            if mirror_state[-3] == 1:
                mirror_state[-3] = 0
            else: mirror_state[-3] = 1
            self.mirror_root = Node(mirror_state)
            self.mirror_root.probabilities = self.root.mirror_probs
            self.mirror_root.build_actions()

            dir_noise1 = np.random.dirichlet(alpha=np.full((15, 1), 0.03).flatten())
            self.root.probabilities = [(1 - epsilon) * prob + epsilon * noise for prob, noise in zip(self.root.probabilities, dir_noise1)]
            dir_noise2 = np.random.dirichlet(alpha=np.full((15, 1), 0.03).flatten())
            self.mirror_root.probabilities = [(1 - epsilon) * prob + epsilon * noise for prob, noise in
                                       zip(self.mirror_root.probabilities, dir_noise2)]


        nodes = np.array([self.root,self.mirror_root])

        while not done:

            white_action = 0
            black_action = 0

            for node in nodes:

                black_player = True if node.state[-3]==0 else False

                actionvalues = [edge.actionvalue for edge in node.edges]
                visits = [edge.visits for edge in node.edges]
                total_visits = sum(visits)
                total_visits_sqrt = math.sqrt(total_visits)

                actions = node.actionspace
                probs = [node.probabilities[i - 1] for i in actions]

                probs = [torch.tensor(1/len(actions)) for i in actions] #works better than learned policy ?

                upper_bounds = [(c * prob.item() * (total_visits_sqrt / (1 + visit))) for prob, visit in zip(probs, visits)]

                if node.state[-3] == 1:
                    upper_bounds = [-bound for bound in upper_bounds]

                edge_values = [q + u for q, u in zip(actionvalues, upper_bounds)]

                if all(x == edge_values[0] for x in edge_values):
                    chosen_edge = random.choice(node.edges)

                    if black_player:
                        black_action = chosen_edge.action

                    else:
                        white_action = chosen_edge.action
                else:

                    if black_player:
                        action_index = edge_values.index(max(edge_values))
                        chosen_edge = node.edges[action_index]
                        black_action = chosen_edge.action

                    else:
                        action_index = edge_values.index(min(edge_values))
                        chosen_edge = node.edges[action_index]
                        white_action = chosen_edge.action

                edge_buffer.append(chosen_edge)

            env_action = self.env.step(black_action,white_action)
            black_score,white_score = self.env.get_player_scores()

            new_node = self.build_node(self.root.state,env_action,black_action,white_action,black_score,white_score)
            new_mirror_node = self.build_node(self.mirror_root.state, env_action, black_action, white_action, black_score,white_score)
            new_roots = [new_node,new_mirror_node]

            root_finder = zip(new_roots,edge_buffer[edge_index:edge_index+2])

            counter = 0
            for root_finder_obj in root_finder:

                contains = False

                for loop_node in root_finder_obj[1].targetNodes:
                    if all(loop_node.state == root_finder_obj[0].state):
                        nodes[counter] = loop_node
                        contains = True
                        counter +=1
                        break

                if not contains:
                    self.set_probabilities(root_finder_obj[0])
                    root_finder_obj[1].targetNodes.append(root_finder_obj[0])
                    nodes[counter] = new_roots[counter]
                    counter +=1
                    done = True
                    child_added = True


            edge_index += 2
            if contains and not child_added:
                done = self.env.check_status()
        self.update_values_double_root(nodes,edge_buffer)






