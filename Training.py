#change buffers into class variables

from Tree import Tree
from Network import Network
from Environment import Environment
from Trajectory import Trajectory
from Node import Node
import torch
import numpy as np
import torch.utils.data.sampler as sampler
import math
import copy


class Training:

    def __init__(self):
        self.dir = r'C:\Users\Abgedrehter Alman\PycharmProjects\bot\model3'
        self.black_root = 0
        self.white_root = 0
        self.black_edge = 0
        self.white_edge = 0
        self.network = Network(49,128)
        self.tree = Tree(self.network)
        self.env = Environment()
        self.trajectory_size = 1000
        self.trajectory = Trajectory(self.trajectory_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0003,weight_decay=1e-5) #L2 weight regularization = weight_decay lr normally 0.0003


    def load_network(self):
        self.network.load_state_dict(torch.load(self.dir))
        self.network.eval()

    def pick_action(self,node):
        temperature = 1.0
        visits = [edge.visits for edge in node.edges]
        total_visits = sum(visits)
        action_probs = [ (math.pow(visit,(1/temperature)) / (math.pow(total_visits,(1/temperature) ))) for visit in visits] #mc_probs
        action_edge = np.random.choice(node.edges , p=action_probs)
        return action_edge , action_probs

    def build_mc_prob(self,mc_prob):
        mc_prob_tensor = torch.zeros([15,1])
        actionspace =  self.tree.root.actionspace
        for action, mc in zip(actionspace,mc_prob):
            mc_prob_tensor[action-1] = mc
        return mc_prob_tensor


    def train_network(self,game_counter):

        if game_counter <=999:
            ub_index = game_counter * 30 - 1
        else: ub_index = 30000

        indeces = list(sampler.BatchSampler(sampler.SubsetRandomSampler(range(ub_index)), batch_size=200, drop_last=False))
        i = 0

        for index in indeces:
            if i >= 6:
                return
            self.optimizer.zero_grad()
            z = self.trajectory.outcome_values[index]
            prob = self.trajectory.mc_probs[index]

            v,net_prob = self.network(self.trajectory.states[index])
            log_prob = torch.log(net_prob)
            lossfn = ((z-v).pow(2) - prob*log_prob).mean()
            lossfn.backward()
            self.optimizer.step()

            i +=1

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

    def build_state(self,env_action,black_score,white_score):
        new_black_state =  copy.copy(self.tree.root.state)
        new_black_state[-1]=white_score
        new_black_state[-2]=black_score
        new_black_state[-3]=0
        new_black_state[env_action+35]=0
        new_black_state[self.black_edge.action-1] = 0
        new_black_state[self.white_edge.action+14] = 0
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
            new_black_root.build_actionspace()
            new_black_root.build_edges()
            self.set_probabilities(new_black_root)
            self.black_root = new_black_root
        if not white:
            new_white_root = Node(new_white_state)
            new_white_root.build_actionspace()
            new_white_root.build_edges()
            self.set_probabilities(new_white_root)
            self.white_root = new_white_root

    def reset(self):
        self.tree.init_tree()
        self.env.full_reset()
        self.tree.env.full_reset()
        self.black_edge = 0
        self.white_edge = 0
        self.black_root = 0
        self.white_root = 0

    def train(self):
        self.load_network()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(device)
        #self.network.to(device)

        training = True
        game_counter = 0
        index = 0
        while training:
            self.reset()
            done = False

            while not done:
                for counter in range (0,400):
                   self.tree.simulation()

                #print(self.tree.root.state)
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
                        new_root.build_actionspace()
                        new_root.build_edges()
                        self.set_probabilities(new_root)

                if state[-3] == 1:
                    self.white_edge = action_edge

                    env_action = self.env.step(self.black_edge.action,self.white_edge.action)
                    black_score, white_score = self.env.get_player_scores()

                    new_black_state,new_white_state = self.build_state(env_action,black_score,white_score)

                    if not all(i==0 for i in new_black_state[0:15]) : #or not all(i==0 for i in new_white_state[15:30]):
                        self.find_roots(new_black_state,new_white_state)
                        new_root = self.black_root

                self.tree.root = copy.copy(new_root)
                self.env.update_env(self.tree.root.state[30:45])

                done = self.env.check_status()

                if done:
                    outcome_value = self.env.eval_game()
                    self.trajectory.add_outcome_values(outcome_value,index)
                    game_counter +=1
                    print("Game: " + str(game_counter) + " played.")
                    if game_counter >= 90:
                        self.train_network(game_counter)
                        torch.save(self.network.state_dict(), self.dir)

                if index>=29999:
                    index = 0
                else:
                    index +=1
