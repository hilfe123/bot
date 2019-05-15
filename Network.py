import torch

class Network(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Network, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.head1 = torch.nn.Linear(hidden_size,15)
        self.head2 = torch.nn.Linear(hidden_size , 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = torch.nn.ReLU()


    def forward(self, x):
        out = self.linear(x)
        out = self.ReLU(out)
        out = self.linear2(out)
        out = self.ReLU(out)
        state_value = self.head2(out)
        state_value = torch.tanh(state_value)
        probabilities = self.head1(out)
        probabilities = self.softmax(probabilities)
        return state_value , probabilities