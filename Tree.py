import numpy as np
from Node import Node
from Environment import Environment
import random
import torch
import copy
import math
#Save alternative game state probs in node

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

    def update_values(self, outcome, edges):
        for edge in edges:
            edge.visits += 1
            edge.totalactionvalue += outcome
            edge.actionvalue = edge.totalactionvalue / edge.visits

    def do_random_actions(self, black_player,node):
        node = node
        self.env.deck = (node.state[30:46])
        while not self.env.check_status():
            if black_player:
                black_actionspace = node.actionspace
                black_action = random.choice(black_actionspace)
                white_actionspace = node.state[15:30]
                white_actionspace = [i for i in white_actionspace if i != 0]
                white_action = random.choice(white_actionspace)
                env_action = self.env.step(black_action, white_action)
                black_score, white_score = self.env.get_player_scores()
                node = self.build_node(node.state, env_action, black_action, white_action, black_score, white_score)
            else:
                white_actionspace = node.actionspace
                white_action = random.choice(white_actionspace)
                black_actionspace = node.state[0:15]
                black_actionspace = [i for i in black_actionspace if i != 0]
                black_action = random.choice(black_actionspace)
                env_action = self.env.step(black_action, white_action)
                black_score, white_score = self.env.get_player_scores()
                node = self.build_node(node.state, env_action, black_action, white_action, black_score, white_score)

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
        node.probabilities = probabilities[0]
        node.mirror_probs = probabilities[1]

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
        c = (len(node.actionspace)/2) * 0.15

        while not done:
            black_player = True if node.state[-3]==0 else False
            actionvalues = [edge.actionvalue for edge in node.edges]

            visits = [edge.visits for edge in node.edges]
            total_visits = sum(visits)
            total_visits_sqrt = math.sqrt(total_visits)

            actions = node.actionspace
            probs = [node.probabilities[i - 1] for i in actions]  # node.probabilities[i - 1] torch.tensor([1])

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
                if all(loop_node.state == new_node.state):
                    edge_buffer.append(chosen_edge)
                    node = loop_node
                    contains = True
                    break

            if not contains and not child_added:
                self.set_probabilities(new_node)
                chosen_edge.targetNodes.append(new_node)
                edge_buffer.append(chosen_edge)
                contains = True
                child_added = True
                node = new_node

            if not contains and child_added:
                self.do_random_actions(black_player,new_node)

            done = self.env.check_status()

        game_outcome = self.env.eval_game()
        self.update_values(game_outcome, edge_buffer)






