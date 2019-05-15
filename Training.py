from Tree import Tree
from Environment import Environment
from Trajectory import Trajectory
from Node import Node
import torch
import numpy as np
import torch.utils.data.sampler as sampler
import math
import copy
from operator import attrgetter
from Network import Network


class Training:

    def __init__(self,dir):
        self.dir = dir

        self.learning_network = Network(49,128)
        self.learning_network.load_state_dict(torch.load(self.dir))
        self.learning_network.eval()
        self.curr_network = Network(49, 128)
        self.curr_network.load_state_dict(torch.load(self.dir))
        self.curr_network.eval()

        self.env = Environment()
        self.trajectory_size = 5000
        self.trajectory = Trajectory(self.trajectory_size)
        self.optimizer = torch.optim.SGD(self.learning_network.parameters(), lr=1e-3,weight_decay=10e-4, momentum=0.9)

        self.tree = Tree(self.curr_network)


    def cross_entropy(self,pred, soft_targets):
        return torch.sum(- soft_targets * torch.log(pred), 1).unsqueeze(dim=0).t()

    def pick_action(self,node):
        if len(node.edges) > 12:
            temperature = 1.0
        else: temperature = 0.01
        visits = [edge.visits for edge in node.edges]
        total_visits = sum(visits)
        action_probs = [ (math.pow(visit,(1/temperature)) / (math.pow(total_visits,(1/temperature) ))) for visit in visits]
        action_probs = np.array(action_probs)
        action_probs /= action_probs.sum()
        action_edge = np.random.choice(node.edges , p=action_probs)
        return action_edge

    def build_mc_prob(self,node):
        mc_prob_tensor = torch.zeros([15,1])
        edge_visits = [edge.visits for edge in node.edges]
        total_visits = sum(edge_visits)
        actions = [edge.action for edge in node.edges]
        actions_and_visits = zip(actions,edge_visits)
        for action,visits in actions_and_visits:
            mc_prob_tensor[action-1] = visits/total_visits
        return mc_prob_tensor

    def train_network(self,game_counter):
        if game_counter <=self.trajectory_size-1:
            ub_index = game_counter * 30 - 1
        else: ub_index = self.trajectory_size*30

        indeces = list(sampler.BatchSampler(sampler.SubsetRandomSampler(range(ub_index)), batch_size=300, drop_last=False))
        i = 0

        for index in indeces:
            if i >= 1:
                return
            self.optimizer.zero_grad()

            z = self.trajectory.outcome_values[index]
            prob = self.trajectory.mc_probs[index]
            v,net_prob = self.learning_network(self.trajectory.states[index])
            cross_entropy_loss =self.cross_entropy(net_prob,prob)

            loss = ((z - v).pow(2) + cross_entropy_loss).mean()
            loss.backward()
            self.optimizer.step()
            i +=1
            print(loss.item())

    def set_probabilities(self, node,network):
        tensor = torch.from_numpy(np.array(node.state)).unsqueeze(dim=0).float()
        prediction_value, probabilities = network(tensor)
        probabilities = probabilities.detach().t()
        prediction_value = prediction_value.detach()
        node.probabilities = [probabilities[action-1] for action in node.actionspace]
        node.v = prediction_value

    def build_states(self,state,black_edge,white_edge,env_action,black_score,white_score):
        new_black_state =  copy.copy(state)
        new_black_state[-1]=white_score
        new_black_state[-2]=black_score
        new_black_state[-3]=0
        new_black_state[env_action+35]=0
        new_black_state[black_edge.action-1] = 0
        new_black_state[white_edge.action+14] = 0

        new_white_state = copy.copy(new_black_state)
        new_white_state[-3]=1
        return new_black_state,new_white_state

    def find_roots(self,new_black_state,new_white_state,black_action_edge,white_action_edge):
        white = False
        black = False
        black_root = 0
        white_root = 0
        for node in black_action_edge.children:
            if all(node.state == new_black_state):
                black_root = node
                black = True
        for node in white_action_edge.children:
            if all(node.state == new_white_state):
                white_root = node
                white = True

        if not black:
            new_black_root = Node(new_black_state)
            new_black_root.build_actions()
            self.set_probabilities(new_black_root, self.curr_network)
            black_root = new_black_root
        if not white:
            new_white_root = Node(new_white_state)
            new_white_root.build_actions()
            self.set_probabilities(new_white_root, self.curr_network)
            white_root = new_white_root

        return black_root,white_root

    def eval_net(self):
        updated_net = 0
        old_net = 0
        draw =0

        for i in range (0,100):

            tree = Tree(self.learning_network)
            tree.init_tree()
            old_tree = Tree(self.curr_network)
            old_tree.init_tree()
            eval_env = Environment()
            done = False

            while not done:
                for x in range (0,400):
                    tree.env.set_env(tree.black_root.state)
                    old_tree.env.set_env(old_tree.black_root.state)
                    tree.double_agent_simulation()
                    old_tree.double_agent_simulation()
                action_edge = max(tree.black_root.edges,key=attrgetter('visits'))
                action = action_edge.action

                old_action_edge = max(old_tree.white_root.edges,key=attrgetter('visits'))
                old_action = old_action_edge.action

                env_action = eval_env.step(action,old_action)
                new,old =eval_env.get_player_scores()
                black_state,white_state = self.build_states(tree.black_root.state,action_edge,old_action_edge,env_action,new,old)

                learning_black_root = Node(black_state)
                learning_black_root.build_actions()
                self.set_probabilities(learning_black_root,self.learning_network)
                learning_white_root = Node(white_state)
                learning_white_root.build_actions()
                self.set_probabilities(learning_white_root,self.learning_network)
                tree.black_root = learning_black_root
                tree.white_root = learning_white_root

                curr_black_root = Node(black_state)
                curr_black_root.build_actions()
                self.set_probabilities(curr_black_root, self.curr_network)
                curr_white_root = Node(white_state)
                curr_white_root.build_actions()
                self.set_probabilities(curr_white_root, self.curr_network)
                old_tree.black_root = curr_black_root
                old_tree.white_root = curr_white_root

                done = eval_env.check_status()
            winner = eval_env.eval_game()
            if winner == 1 :
                updated_net +=1
                print("Net won this game " + str(updated_net) )
            elif winner == 0:
                draw +=1
                print("Draw " + str(draw))
            else:
                old_net +=1
                print("Old Net won. " + str(old_net))

        print("New Network won " + str(updated_net) + " matches against the old one.")
        print("Draws: "+str(draw))

        if updated_net / (updated_net + old_net) >= 0.54:
            torch.save(self.learning_network.state_dict(), self.dir)
            self.curr_network.load_state_dict(torch.load(self.dir))
            self.curr_network.eval()
            print("---Network updated---")
        else:
            self.learning_network.load_state_dict(torch.load(self.dir))
            self.learning_network.eval()

    def load_dependencies(self):
        self.trajectory.states = torch.load("models and trajectory/trajectory_states.pt")
        self.trajectory.mc_probs = torch.load("models and trajectory/trajectory_mc_probs.pt")
        self.trajectory.outcome_values = torch.load("models and trajectory/trajectory_outcome_values.pt")
        game_counter = torch.load("models and trajectory/training_gamecounter.pt")
        index = torch.load("models and trajectory/training_index.pt")
        return game_counter.item(),index.item()

    def save_dependencies(self,game_counter,index):
        torch.save(self.trajectory.states, "models and trajectory/trajectory_states.pt")
        torch.save(self.trajectory.mc_probs, "models and trajectory/trajectory_mc_probs.pt")
        torch.save(self.trajectory.outcome_values, "models and trajectory/trajectory_outcome_values.pt")
        torch.save(torch.tensor(game_counter), "models and trajectory/training_gamecounter.pt")
        torch.save(torch.tensor(index), "models and trajectory/training_index.pt")


    def train(self):
        training = True
        game_counter , index = self.load_dependencies()
        while training:
            self.tree.init_tree()
            self.tree.setNetwork(self.curr_network)
            self.env.full_reset()
            done = False
            while not done:
                self.tree.noise = False

                for counter in range(0,400):
                    self.tree.env.set_env(self.tree.black_root.state)
                    self.tree.double_agent_simulation()

                black_action_edge  = self.pick_action(self.tree.black_root)
                white_action_edge  = self.pick_action(self.tree.white_root)
                black_mc_prob = self.build_mc_prob(self.tree.black_root)
                white_mc_prob = self.build_mc_prob(self.tree.white_root)
                self.trajectory.insert(black_mc_prob,copy.copy(self.tree.black_root.state),index)
                self.trajectory.insert(white_mc_prob, copy.copy(self.tree.white_root.state), index+1)

                env_action = self.env.step(black_action_edge.action,white_action_edge.action)
                black_score , white_score = self.env.get_player_scores()
                black_state, white_state = self.build_states(self.tree.black_root.state,black_action_edge,white_action_edge,env_action,black_score,white_score)
                black_root , white_root = self.find_roots(black_state,white_state,black_action_edge,white_action_edge)
                self.tree.black_root = black_root
                self.tree.white_root = white_root

                done = self.env.check_status()

                if index>=(self.trajectory_size*30)-2:
                    index = 0
                else:
                    index +=2

                if done:
                    outcome_value = self.env.eval_game()
                    self.trajectory.add_outcome_values(outcome_value,index)
                    game_counter += 1
                    print("Game: " + str(game_counter) + " played.")
                    if game_counter >= 501:
                        self.train_network(game_counter)

                        if game_counter%500 == 0 :
                            self.save_dependencies(game_counter,index)
                            #self.eval_net()
                            torch.save(self.learning_network.state_dict(), self.dir)
                            self.curr_network.load_state_dict(torch.load(self.dir))
                            self.curr_network.eval()

