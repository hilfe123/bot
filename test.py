import torch
from Network import Network
import numpy as np
#network3 = Network(49,128)

#network3.load_state_dict(torch.load(r'C:\Users\Abgedrehter Alman\PycharmProjects\AGZ\model3'))
#network3.eval()
#array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,0,0,0])
#tensor = torch.from_numpy(array).float()
#torch.save(network3.state_dict(), r'C:\Users\Abgedrehter Alman\PycharmProjects\bot\model3')

#,probs2 = network3(tensor)
#print("Version3")
#print(v2.item())
#print(probs2)
tensor = torch.Tensor(3,3)
tensor[0] = torch.Tensor([3,3,3])