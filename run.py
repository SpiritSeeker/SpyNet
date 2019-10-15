from neurons import Neuron
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from time import time

n = Neuron(0,0)

ini = []
inv = []

start = time()
for i in range(50000):
	inv.append(deepcopy(n.simulate_step(0.25)))
	ini.append(deepcopy(n.soma.membrane_potential))
print(time() - start)
invs = np.asarray(inv)
inis = np.asarray(ini)
plt.plot(invs)
plt.plot(inis)
plt.show()