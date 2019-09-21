import numpy as np
import matplotlib.pyplot as plt

class MLE_Neuron(object):
	"""docstring for MLE_Neuron"""
	def __init__(self, params = {'gca':4.4,'gk':8,'gl':2,'eca':120,'ek':-84,'el':-60,'phi':0.02,'V1':-1.2,'V2':18,'V3':2,'V4':30,'V5':2,'V6':30,'C':20}):
		super(MLE_Neuron, self).__init__()
		self.params = params
	
	def m_inf(self, v):
		return 0.5 * (1 + np.tanh((v-self.params['V1'])/self.params['V2']))	

	def w_inf(self, v):
		return 0.5 * (1 + np.tanh((v-self.params['V3'])/self.params['V4']))	

	def tau_w(self, v):
		return 1 / (np.cosh((v-self.params['V5'])/self.params['V6']))

	def plot_null_clines(self, i_ext):
		self.get_eq_points(i_ext)
		plt.plot(self.ncs['v'],self.ncs['v_null_cline'],self.ncs['v'],self.ncs['w_null_cline'],self.eq_points[0],self.eq_points[1],'go')
		plt.axis([np.min(self.ncs['v']), np.max(self.ncs['v']), 0, 1])
		plt.show()

	def get_eq_points(self, i_ext):
		v_lower = min(self.params['ek'],self.params['el'],self.params['eca'])
		v_upper = max(self.params['ek'],self.params['el'],self.params['eca'])
		v = np.linspace(v_lower, v_upper, (v_upper-v_lower)/0.0001, endpoint=True)
		v_nc = (i_ext - (self.params['gca']*self.m_inf(v)*(v-self.params['eca']))-(self.params['gl']*(v-self.params['el']))) / (np.finfo(float).eps+self.params['gk']*(v-self.params['ek']))
		w_nc = self.w_inf(v)
		self.ncs = {'v':v, 'v_null_cline':v_nc, 'w_null_cline':w_nc}
		idx = np.argwhere(np.diff(np.sign(self.ncs['v_null_cline']-self.ncs['w_null_cline']))).flatten()
		self.eq_points = [self.ncs['v'][idx],self.ncs['w_null_cline'][idx]]

	def get_dvdt(self, w, v, i_ext=0):
		return (i_ext - (self.params['gca']*self.m_inf(v)*(v-self.params['eca'])) - (self.params['gk']*w*(v-self.params['ek'])) - (self.params['gl']*(v-self.params['el']))) / self.params['C']

	def get_dwdt(self, w, v):
		return (self.params['phi']*(self.w_inf(v)-w)) / self.tau_w(v)
