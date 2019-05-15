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
        self.black_root = 0
        self.white_root = 0
        self.env = Environment()
        self.noise = False

    def setNetwork(self,network):
        self.network = network

    def init_tree(self):
        self.black_root = Node(
            np.array([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0])
        )
        self.black_root.build_actions()
        self.set_probabilities(self.black_root)

        self.white_root = Node(np.array([
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 0, 0])
        )
        self.white_root.build_actions()
        self.set_probabilities(self.white_root)


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
        tensor = torch.from_numpy(np.array(node.state)).unsqueeze(dim=0).float()
        prediction_value, probabilities = self.network(tensor)
        probabilities = probabilities.detach().t()
        prediction_value = prediction_value.detach()
        node.probabilities = [ probabilities[action-1].item() for action in node.actionspace]
        node.v = prediction_value


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


    def double_agent_simulation(self):

        child_added = False
        done = False
        edge_buffer = []
        edge_index = 0
        c = 3
        dir_dist = 0.09
        epsilon = 0.25

        if not self.noise:
            dir_noise1 = np.random.dirichlet(alpha=np.full((len(self.black_root.actionspace), 1), dir_dist).flatten())
            self.black_root.probabilities = [(1 - epsilon) * prob + epsilon * noise for prob, noise in zip(self.black_root.probabilities, dir_noise1)]
            dir_noise2 = np.random.dirichlet(alpha=np.full((len(self.white_root.actionspace), 1), dir_dist).flatten())
            self.white_root.probabilities = [(1 - epsilon) * prob + epsilon * noise for prob, noise in zip(self.white_root.probabilities, dir_noise2)]
            self.noise = True

        nodes = np.array([self.black_root,self.white_root])

        while not done:

            white_action = 0
            black_action = 0

            for node in nodes:

                black_player = True if node.state[-3]==0 else False

                actionvalues = [edge.actionvalue for edge in node.edges]
                visits = [edge.visits for edge in node.edges]
                total_visits = sum(visits)
                total_visits_sqrt = math.sqrt(total_visits)

                probs = node.probabilities

                upper_bounds = [(c * prob * (total_visits_sqrt / (1 + visit))) for prob, visit in zip(probs, visits)]

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

            new_black_node = self.build_node(self.black_root.state,env_action,black_action,white_action,black_score,white_score)
            new_white_node = self.build_node(self.white_root.state, env_action, black_action, white_action, black_score,white_score)
            new_roots = [new_black_node,new_white_node]

            root_finder = zip(new_roots,edge_buffer[edge_index:edge_index+2])

            counter = 0
            for root_finder_obj in root_finder:

                contains = False
                if not child_added:
                    for loop_node in root_finder_obj[1].children:
                        if all(loop_node.state == root_finder_obj[0].state):
                            nodes[counter] = loop_node
                            contains = True
                            counter +=1
                            break

                if not contains:
                    self.set_probabilities(root_finder_obj[0])
                    root_finder_obj[1].children.append(root_finder_obj[0])
                    nodes[counter] = new_roots[counter]
                    counter +=1
                    done = True
                    child_added = True


            edge_index += 2
            if contains and not child_added:
                done = self.env.check_status()
        self.update_values_double_root(nodes,edge_buffer)

    def stochastic_tree_search(self):
        child_added = False
        done = False
        edge_buffer = []
        edge_index = 0
        c = 0.95
        dir_dist = 0.09
        epsilon = 0.25

        if not self.noise:
            dir_noise1 = np.random.dirichlet(alpha=np.full((len(self.black_root.actionspace), 1), dir_dist).flatten())
            self.black_root.probabilities = [(1 - epsilon) * prob + epsilon * noise for prob, noise in
                                             zip(self.black_root.probabilities, dir_noise1)]
            dir_noise2 = np.random.dirichlet(alpha=np.full((len(self.white_root.actionspace), 1), dir_dist).flatten())
            self.white_root.probabilities = [(1 - epsilon) * prob + epsilon * noise for prob, noise in
                                             zip(self.white_root.probabilities, dir_noise2)]
            self.noise = True

        nodes = np.array([self.black_root, self.white_root])

        while not done:

            white_action = 0
            black_action = 0

            for node in nodes:

                black_player = True if node.state[-3] == 0 else False

                actionvalues = [edge.actionvalue for edge in node.edges]
                visits = [edge.visits for edge in node.edges]
                total_visits = sum(visits)
                total_visits_sqrt = math.sqrt(total_visits)

                probs = node.probabilities
                #probs = [1/len(probs) for prob in probs]

                upper_bounds = [(c * prob * (total_visits_sqrt / (1 + visit))) for prob, visit in zip(probs, visits)]

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

            env_action = self.env.step(black_action, white_action)
            black_score, white_score = self.env.get_player_scores()

            new_black_node = self.build_node(self.black_root.state, env_action, black_action, white_action, black_score,
                                             white_score)
            new_white_node = self.build_node(self.white_root.state, env_action, black_action, white_action, black_score,
                                             white_score)
            new_roots = [new_black_node, new_white_node]

            root_finder = zip(new_roots, edge_buffer[edge_index:edge_index + 2])

            counter = 0
            for root_finder_obj in root_finder:

                contains = False
                if not child_added:
                    for loop_node in root_finder_obj[1].children:
                        if all(loop_node.state[0:30] == root_finder_obj[0].state[0:30]):
                            self.set_probabilities(root_finder_obj[0])
                            nodes[counter] = new_roots[counter]
                            contains = True
                            counter += 1
                            break

                if not contains:
                    self.set_probabilities(root_finder_obj[0])
                    root_finder_obj[1].children.append(root_finder_obj[0])
                    nodes[counter] = new_roots[counter]
                    counter += 1
                    done = True
                    child_added = True

            edge_index += 2
            if contains and not child_added:
                done = self.env.check_status()
        self.stochastic_value_updates(black_action,white_action,env_action,edge_buffer)


    def stochastic_value_updates(self,black_action,white_action,env_action,edge_buffer):

        possible_env_actions = [action for action in self.env.deck if action != 0]
        possible_env_actions.append(env_action)

        black_state = copy.copy(self.black_root.state)
        black_state[black_action-1] = 0
        black_state[white_action+14] = 0

        white_state = copy.copy(self.white_root.state)
        white_state[black_action+1] = 0
        white_state[white_action+14] =0


        black_tensor = torch.zeros(len(possible_env_actions),49)
        white_tensor = torch.zeros(len(possible_env_actions),49)
        index = 0

        for env_action in possible_env_actions:
            state = copy.copy(black_state)
            state[env_action + 35] = 0
            if env_action > 0 and black_action > white_action:
                state[-2] += env_action
            if env_action < 0 and black_action > white_action:
                state[-1] += env_action
            if env_action > 0 and black_action < white_action:
                state[-1] += env_action
            if env_action < 0 and black_action < white_action:
                state[-2] += env_action
            black_tensor[index] = torch.tensor(state)
            index +=1

        index = 0
        for env_action in possible_env_actions:
            state = copy.copy(white_state)
            state[env_action + 35] = 0
            if env_action > 0 and black_action > white_action:
                state[-2] += env_action
            if env_action < 0 and black_action > white_action:
                state[-1] += env_action
            if env_action > 0 and black_action < white_action:
                state[-1] += env_action
            if env_action < 0 and black_action < white_action:
                state[-2] += env_action
            white_tensor[index] = torch.tensor(state)
            index +=1
        black_prediction_value, probabilities = self.network(black_tensor)
        white_prediction_value, probabilities = self.network(white_tensor)
        black_prediction_value = black_prediction_value.detach()
        white_prediction_value = white_prediction_value.detach()
        black_v = black_prediction_value.mean()
        white_v = white_prediction_value.mean()
        v = [black_v,white_v]
        for edge in edge_buffer:

            if edge_buffer.index(edge)%2 == 0:
                index = 0
            else: index =1

            edge.visits +=1
            edge.totalactionvalue += v[index].item()
            edge.actionvalue = edge.totalactionvalue / edge.visits

