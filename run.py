from neurons import Neuron, nonNMDA_Synapse
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from time import time

n = Neuron(0,1,[0.1],1,[5])
syn = nonNMDA_Synapse(n.soma.membrane_potential)

ini = []
inv = []

start = time()
for i in range(50000):
	vs = n.simulate_step([1])
	inv.append(deepcopy(syn.simulate_step(vs[0])))
	ini.append(deepcopy(vs[0]))
print(time() - start)
invs = np.asarray(inv)
inis = np.asarray(ini)
plt.plot(invs)
plt.plot(inis)
plt.show()