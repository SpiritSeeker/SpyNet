import numpy as np
import neurons
import currents
from copy import deepcopy
from time import time

class SpyNet(object):
	"""docstring for SpyNet"""
	def __init__(self):
		self.neurons_list = []
		self.synapse_dict_list = []
		self.n_index = 0
		self.syn_index = 0
		self.axon_terminal_voltages = []
		self.postsynaptic_conductances = []
		self.simulated = []

	def addNeuron(self, n_in = 1, in_dists = None, n_out = 1, out_delays = None):
		self.neurons_list.append(neurons.Neuron(self.n_index, n_in, in_dists, n_out, out_delays))
		axon_terminal_voltages = [a.terminal_voltage for a in self.neurons_list[self.n_index].axons]
		self.axon_terminal_voltages.append(axon_terminal_voltages)
		postsynaptic_conductances = n_in * [0]
		self.postsynaptic_conductances.append(postsynaptic_conductances)
		self.simulated.append(False)
		self.n_index += 1
		return (self.n_index-1)

	def connect(self, presyn_id, postsyn_id, syn_type):
		if presyn_id[0] >= self.n_index:
			print('Error: Neuron with id \''+str(presyn_id[0])+'\' does not exist.')
			return -1
		if postsyn_id[0] >= self.n_index:
			print('Error: Neuron with id \''+str(postsyn_id[0])+'\' does not exist.')
			return -1
		if presyn_id[1] >= len(self.neurons_list[presyn_id[0]].axons):
			print('Error: Neuron \''+str(presyn_id[0])+'\' does not have an axon with id \''+str(presyn_id[1])+'\'.')
			return -1
		if postsyn_id[1] >= len(self.neurons_list[postsyn_id[0]].dendrites):
			print('Error: Neuron \''+str(postsyn_id[0])+'\' does not have a dendrite with id \''+str(postsyn_id[1])+'\'.')
			return -1
		synapse_dict = {}
		if syn_type == 'non-nmda':
			synapse_dict['synapse'] = neurons.nonNMDA_Synapse()
			synapse_dict['syn_type'] = 'non-nmda'
		elif syn_type == 'nmda':
			synapse_dict['synapse'] = neurons.NMDA_Synapse()
			synapse_dict['syn_type'] = 'nmda'	
		synapse_dict['presyn_id'] = presyn_id
		synapse_dict['postsyn_id'] = postsyn_id
		self.synapse_dict_list.append(synapse_dict)	
		self.syn_index += 1
		return (self.syn_index-1)

	def simulate_step(self, inp_index, inps, timestep = 1e-3):
		for i in range(len(inp_index)):
			v_terminals = self.neurons_list[inp_index[i]].simulate_step(inps[i])
			self.axon_terminal_voltages[inp_index[i]] = v_terminals
			self.simulated[inp_index[i]] = True

		for i in range(self.n_index):
			if not self.simulated[i]:
				v_terminals = self.neurons_list[i].simulate_step(self.postsynaptic_conductances[i])
				self.axon_terminal_voltages[i] = v_terminals
			else:
				self.simulated[i] = False

		for s in self.synapse_dict_list:
			if s['syn_type'] == 'non-nmda':
				self.postsynaptic_conductances[s['postsyn_id'][0]][s['postsyn_id'][1]] = s['synapse'].simulate_step(self.axon_terminal_voltages[s['presyn_id'][0]][s['presyn_id'][1]])
			elif s['syn_type'] == 'nmda':
				self.postsynaptic_conductances[s['postsyn_id'][0]][s['postsyn_id'][1]] = s['synapse'].simulate_step(self.axon_terminal_voltages[s['presyn_id'][0]][s['presyn_id'][1]], self.neurons_list[s['postsyn_id'][0]].dendrites[s['postsyn_id'][1]].vms[1])

	def simulate(self, inp_index, inps, end_time = 100, timestep = 1e-3):
		start_time = time()
		for i in range(len(inps)):
			for j in range(len(inps[i])):
				if isinstance(inps[i][j], currents.CInput):
					inps[i][j].reset()
				elif isinstance(inps[i][j], int) or isinstance(inps[i][j], float):
					val = inps[i][j]
					inps[i][j] = currents.CInput()
					inps[i][j].add(currents.CStep(val, timestep = timestep))

		total_steps = int(end_time/timestep)
		membrane_potentials = np.zeros([self.n_index, total_steps+1])
		t_s = np.linspace(0, end_time, num = total_steps + 1)
		
		for i in range(self.n_index):
			membrane_potentials[i,0] = self.neurons_list[i].soma.membrane_potential
		
		for t in range(1,total_steps+1):
			self.simulate_step(inp_index, inps, timestep = timestep)
			for i in range(self.n_index):
				membrane_potentials[i,t] = self.neurons_list[i].soma.membrane_potential

		print("Time taken: " + "{0:.2f}".format(time()-start_time) + "s")	
		return deepcopy(membrane_potentials), t_s	
	
	def reset(self):
		for n in self.neurons_list:
			n.reset()
		for s in self.synapse_dict_list:
			s['synapse'].reset()

	def clear(self):
		self.neurons_list = []
		self.synapse_dict_list = []
		self.n_index = 0
		self.syn_index = 0
		self.axon_terminal_voltages = []
		self.postsynaptic_conductances = []
		self.simulated = []