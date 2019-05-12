from Training import Training
from Network import Network
import torch

network1 = Network(49,128)


dir1 = r"C:\Users\Abgedrehter Alman\PycharmProjects\bot\model"

network1.load_state_dict(torch.load(dir1))
network1.eval()

#3 Layers 128 Neurons
training = Training(network1,dir1,True)
training.train()
