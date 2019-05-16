from Training import Training
from Network import Network

network1 = Network(49,128)

dir1 = r"C:\Users\Abgedrehter Alman\PycharmProjects\bot\models and trajectory\model3"

training = Training(dir1)
training.testing()