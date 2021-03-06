from Edge import Edge

class Node:
    def __init__(self,state):
        self.state = state
        self.actionspace = 0
        self.edges = []
        self.probabilities = []
        self.mirror_probs = []
        self.v = 0

    def build_actions(self):
        if self.state[-3] == 0:
            self.actionspace = self.state[0:15]
            self.actionspace = [value for value in self.actionspace if value != 0]
        if self.state[-3] == 1:
            self.actionspace = self.state[15:30]
            self.actionspace = [value for value in self.actionspace if value != 0]
        for action in self.actionspace:
            edge = Edge()
            edge.action = action
            self.edges.append(edge)