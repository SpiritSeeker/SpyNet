from neurons import Neuron, nonNMDA_Synapse
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from time import time
from network import SpyNet

s_net = SpyNet()

s_net.addNeuron(1, [0.1], 1, [5])
s_net.addNeuron(1, [0.11], 1, [7])
s_net.addNeuron(2, [0.17, 0.06], 2, [10, 7])
s_net.addNeuron(1, [0.05], 1, [5])
s_net.addNeuron(1, [0.07], 1, [5])

s_net.connect([0,0],[2,0],'non-nmda')
s_net.connect([1,0],[2,1],'non-nmda')
s_net.connect([2,0],[3,0],'non-nmda')
s_net.connect([2,1],[4,0],'non-nmda')

s1 = []
s2 = []
s3 = []
s4 = []
s5 = []
gs1 = []
gs2 = []

start = time()
for i in range(50000):
    s_net.simulate_step([0,1],[[0.1],[0.2]])
    s1.append(s_net.neurons_dict['0'].soma.membrane_potential)
    s2.append(s_net.neurons_dict['1'].soma.membrane_potential)
    s3.append(s_net.neurons_dict['2'].soma.membrane_potential)
    s4.append(s_net.neurons_dict['3'].soma.membrane_potential)
    s5.append(s_net.neurons_dict['4'].soma.membrane_potential)
    gs1.append(s_net.postsynaptic_conductances[2][0])
    gs2.append(s_net.postsynaptic_conductances[2][1])
print(time() - start)

s1s = np.asarray(s1)
s2s = np.asarray(s2)
s3s = np.asarray(s3)
s4s = np.asarray(s4)
s5s = np.asarray(s5)
gs1s = np.asarray(gs1)
gs2s = np.asarray(gs2)

plt.plot(s1s)
plt.plot(s2s)
plt.plot(s3s)
plt.plot(s4s)
plt.plot(s5s)
plt.show()