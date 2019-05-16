import torch
from Network import Network
import numpy as np



network2 = Network(49,128)
network2.load_state_dict(torch.load(r'C:\Users\User\PycharmProjects\bot\models and trajectory\model3'))
network2.eval()
array = np.zeros((2,49))
array[0] = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,0,0,0])
array[1] = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,1,0,0])
array[0] = np.array([0,0,0,0,5,6,7,0,0,0,0,0,0,0,0,0,0,0,0,5,6,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,6,7,0,0,0,0,0,0])
array[1] = np.array([0,0,0,0,5,6,7,0,0,0,0,0,0,0,0,0,0,0,0,5,6,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,6,7,0,0,0,1,0,0])
#array[0] = np.array([[ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,15, 13]])

tensor = torch.from_numpy(array).float()

v2,probs2 = network2(tensor)

print(v2)
print(probs2)
