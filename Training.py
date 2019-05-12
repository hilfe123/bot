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

    def __init__(self,network, dir, double_root_tree):
        self.dir = dir
        self.double_root_tree = double_root_tree
        self.black_root = 0
        self.white_root = 0
        self.black_edge = 0
        self.white_edge = 0
        self.network = network
        self.env = Environment()
        self.trajectory_size = 3500
        self.trajectory = Trajectory(self.trajectory_size)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=1e-3,weight_decay=1e-4, momentum=0.9)
        self.backup_network = Network(49,128)
        self.backup_network.load_state_dict(torch.load(self.dir))
        self.backup_network.eval()
        self.tree = Tree(self.backup_network)

    def cross_entropy(self,pred, soft_targets):
        return torch.sum(- soft_targets * torch.log(pred), 1).unsqueeze(dim=0).t()

    def pick_action(self,node):
        if len(node.edges) > 3:
            temperature = 1.0
        else: temperature = 0.01
        visits = [edge.visits for edge in node.edges]
        total_visits = sum(visits)
        action_probs = [ (math.pow(visit,(1/temperature)) / (math.pow(total_visits,(1/temperature) ))) for visit in visits]
        action_probs = np.array(action_probs)
        action_probs /= action_probs.sum()

        action_edge = np.random.choice(node.edges , p=action_probs)
        return action_edge , action_probs

    def build_mc_prob(self,mc_prob):
        mc_prob_tensor = torch.zeros([15,1])
        actionspace =  self.tree.root.actionspace
        for action, mc in zip(actionspace,mc_prob):
            mc_prob_tensor[action-1] = mc
        return mc_prob_tensor


    def train_network(self,game_counter):
        if game_counter <=self.trajectory_size-1:
            ub_index = game_counter * 30 - 1
        else: ub_index = self.trajectory_size*30

        indeces = list(sampler.BatchSampler(sampler.SubsetRandomSampler(range(ub_index)), batch_size=75, drop_last=False)) #75,3

        i = 0
        for index in indeces:
            if i >= 3:
                return
            self.optimizer.zero_grad()
            z = self.trajectory.outcome_values[index]
            prob = self.trajectory.mc_probs[index]

            v,net_prob = self.network(self.trajectory.states[index])
            cross_entropy_loss =self.cross_entropy(net_prob,prob)
            lossfn = ((z - v).pow(2) + cross_entropy_loss).mean()
            lossfn.backward()
            self.optimizer.step()
            i +=1
            print(lossfn.item())

    def set_probabilities(self, node , network):
        mirror_state = copy.copy(node.state)
        if node.state[-3] == 1:
            mirror_state[-3] = 0
        else: mirror_state[-3] = 1
        tensor = torch.zeros(2,49)
        tensor[0] = torch.from_numpy(np.array(node.state)).float()
        tensor[1] = torch.from_numpy(np.array(mirror_state)).float()
        prediction_value, probabilities = network(tensor)
        probabilities = probabilities.detach()
        prediction_value = prediction_value.detach()
        node.probabilities = probabilities[0]
        node.mirror_probs = probabilities[1]
        node.v = prediction_value[0]

    def build_state(self,state,black_edge,white_edge,env_action,black_score,white_score):
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

    def find_roots(self,new_black_state,new_white_state):
        white = False
        black = False
        for node in self.black_edge.targetNodes:
            if all(node.state == new_black_state):
                self.black_root = node
                black = True
        for node in self.white_edge.targetNodes:
            if all(node.state == new_white_state):
                self.white_root = node
                white = True

        if not white or not black:
            self.build_new_roots(new_black_state,new_white_state,black,white)

    def build_new_roots(self,new_black_state,new_white_state,black,white):
        if not black:
            new_black_root = Node(new_black_state)
            new_black_root.build_actions()
            self.set_probabilities(new_black_root,self.network)
            self.black_root = new_black_root
        if not white:
            new_white_root = Node(new_white_state)
            new_white_root.build_actions()
            self.set_probabilities(new_white_root,self.network)
            self.white_root = new_white_root

    def reset(self):
        self.tree.init_tree()
        self.env.full_reset()
        self.tree.env.full_reset()
        self.black_edge = 0
        self.white_edge = 0
        self.black_root = 0
        self.white_root = 0
        self.tree.setNetwork(self.backup_network)

    def eval_net(self):

        updated_net = 0
        old_net = 0
        draw =0

        for i in range (0,100):

            tree = Tree(self.network)
            tree.init_tree()
            old_tree = Tree(self.backup_network)
            old_tree.init_tree()
            eval_env = Environment()
            done = False
            while not done:
                for x in range (0,400):
                    tree.simulation()
                    old_tree.simulation()
                action_edge = max(tree.root.edges,key=attrgetter('visits'))
                action = action_edge.action

                old_action_edge = max(old_tree.root.edges,key=attrgetter('visits'))
                old_action = old_action_edge.action


                env_action = eval_env.step(action,old_action)
                new,old =eval_env.get_player_scores()
                state,altstate = self.build_state(tree.root.state,action_edge,old_action_edge,env_action,new,old)

                new_root = Node(state)
                new_root.build_actions()
                self.set_probabilities(new_root,self.network)
                tree.root = new_root

                old_root = Node(altstate)
                old_root.build_actions()
                self.set_probabilities(old_root,self.backup_network)
                old_tree.root = old_root

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
            torch.save(self.network.state_dict(), self.dir)
            self.backup_network.load_state_dict(torch.load(self.dir))
            self.backup_network.eval()
            print("---Network updated---")
        else:
            self.network.load_state_dict(torch.load(self.dir))
            self.network.eval()

    def load_dependencies(self):
        self.trajectory.states = torch.load("trajectory_states.pt")
        self.trajectory.mc_probs = torch.load("trajectory_mc_probs.pt")
        self.trajectory.outcome_values = torch.load("trajectory_outcome_values.pt")
        game_counter = torch.load("training_gamecounter.pt")
        index = torch.load("training_index.pt")
        return game_counter.item(),index.item()

    def save_dependencies(self,game_counter,index):
        torch.save(self.trajectory.states, "trajectory_states.pt")
        torch.save(self.trajectory.mc_probs, "trajectory_mc_probs.pt")
        torch.save(self.trajectory.outcome_values, "trajectory_outcome_values.pt")
        torch.save(torch.tensor(game_counter), "training_gamecounter.pt")
        torch.save(torch.tensor(index), "training_index.pt")


    def train(self):

        training = True
        game_counter , index = self.load_dependencies()
        #game_counter = 0
        #index = 0
        while training:
            self.reset()
            done = False

            while not done:
                if not self.double_root_tree:
                    for counter in range (0,400):
                        self.tree.simulation()
                else:
                    self.tree.mirror_root = 0
                    for counter in range(0,400):

                        self.tree.double_root_simulation()

                action_edge ,mc_prob = self.pick_action(self.tree.root)
                mc_prob = self.build_mc_prob(mc_prob)
                self.trajectory.insert(mc_prob,copy.copy(self.tree.root.state),index)

                state = copy.copy(self.tree.root.state)

                if state[-3] == 0:
                    self.black_edge = action_edge
                    self.black_root = copy.copy(self.tree.root)
                    if self.white_root != 0:
                        new_root = self.white_root
                    else:
                        white_state = self.tree.root.state
                        white_state[-3] = 1
                        new_root = Node(white_state)
                        new_root.build_actions()
                        self.set_probabilities(new_root,self.network)

                if state[-3] == 1:
                    self.white_edge = action_edge

                    env_action = self.env.step(self.black_edge.action,self.white_edge.action)
                    black_score, white_score = self.env.get_player_scores()

                    new_black_state,new_white_state = self.build_state(self.tree.root.state,self.black_edge,self.white_edge,env_action,black_score,white_score)

                    if not all(i==0 for i in new_black_state[0:15]) :
                        self.find_roots(new_black_state,new_white_state)
                        new_root = self.black_root

                self.tree.root = copy.copy(new_root)
                self.env.update_env(self.tree.root.state[30:45])

                done = self.env.check_status()

                if done:

                    outcome_value = self.env.eval_game()
                    self.trajectory.add_outcome_values(outcome_value,index)
                    game_counter += 1
                    print("Game: " + str(game_counter) + " played.")
                    if game_counter >= 501 and game_counter%1 == 0:
                        self.train_network(game_counter)

                        if game_counter%500 ==0 :
                            self.save_dependencies(game_counter,index)
                            self.eval_net()


                if index>=(self.trajectory_size*30) -1:
                    index = 0
                else:
                    index +=1
