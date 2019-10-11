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
	a = deepcopy(n.simulate_step(0.25))
	ini.append(a[0])
	inv.append(a[1])
print(time() - start)
invs = np.asarray(inv)
plt.plot(invs[:,-1] + n.soma.eq_points['v_eq'])
plt.show()