import numpy as np
import neurons
import currents

class SpyNet(object):
	"""docstring for SpyNet"""
	def __init__(self):
		self.neurons_dict = {}
		self.synapse_dict_list = []
		self.n_index = 0
		self.syn_index = 0
		self.axon_terminal_voltages = []
		self.postsynaptic_conductances = []
		self.simulated = []

	def addNeuron(self, n_in = 1, in_dists = None, n_out = 1, out_delays = None):
		self.neurons_dict[str(self.n_index)] = neurons.Neuron(self.n_index, n_in, in_dists, n_out, out_delays)
		axon_terminal_voltages = [a.terminal_voltage for a in self.neurons_dict[str(self.n_index)].axons]
		self.axon_terminal_voltages.append(axon_terminal_voltages)
		postsynaptic_conductances = n_in * [0]
		self.postsynaptic_conductances.append(postsynaptic_conductances)
		self.simulated.append(False)
		self.n_index += 1
		return (self.n_index-1)

	def connect(self, presyn_id, postsyn_id, syn_type):
		synapse_dict = {}
		if syn_type == 'non-nmda':
			synapse_dict['synapse'] = neurons.nonNMDA_Synapse()
		synapse_dict['presyn_id'] = presyn_id
		synapse_dict['postsyn_id'] = postsyn_id
		self.synapse_dict_list.append(synapse_dict)	
		self.syn_index += 1
		return (self.syn_index-1)

	def simulate_step(self, inp_index, inps, timestep = 1e-3):
		for i in range(len(inp_index)):
			v_terminals = self.neurons_dict[str(inp_index[i])].simulate_step(inps[i])
			self.axon_terminal_voltages[inp_index[i]] = v_terminals
			self.simulated[inp_index[i]] = True

		for i in range(self.n_index):
			if not self.simulated[i]:
				v_terminals = self.neurons_dict[str(i)].simulate_step(self.postsynaptic_conductances[i])
				self.axon_terminal_voltages[i] = v_terminals
			else:
				self.simulated[i] = False

		for s in self.synapse_dict_list:
			self.postsynaptic_conductances[s['postsyn_id'][0]][s['postsyn_id'][1]] = s['synapse'].simulate_step(self.axon_terminal_voltages[s['presyn_id'][0]][s['presyn_id'][1]])