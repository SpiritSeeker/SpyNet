import numpy as np
import neurons
import currents

class SpyNet(object):
	"""docstring for SpyNet"""
	def __init__(self):
		self.neurons_dict = {}
		self.index = 0

	def addNeuron(self, n_in = 1, in_dists = None, n_out = 1, out_delays = None):
		self.neurons_dict[str(self.index)] = neurons.Neuron(self.index, n_in, in_dists, n_out, out_delays)
		self.index += 1
		return (self.index-1)

	def connect(self, in_id, out_id):
		pass