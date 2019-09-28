import numpy as np
import matplotlib.pyplot as plt
import currents

class MLE_Neuron(object):
	"""docstring for MLE_Neuron"""
	def __init__(self, params = {'gca':4.4,'gk':8,'gl':2,'eca':120,'ek':-84,'el':-60,'phi':0.02,'V1':-1.2,'V2':18,'V3':2,'V4':30,'V5':2,'V6':30,'C':20}):
		self.params = params
		self.get_eq_points(0)
		self.stable = True
	
	def m_inf(self, v):
		return 0.5 * (1 + np.tanh((v-self.params['V1'])/self.params['V2']))	

	def w_inf(self, v):
		return 0.5 * (1 + np.tanh((v-self.params['V3'])/self.params['V4']))	

	def tau_w(self, v):
		return 1 / (np.cosh((v-self.params['V5'])/self.params['V6']))

	def plot_null_clines(self, i_ext):
		if i_ext is not 0:
			self.get_eq_points(i_ext)
		plt.plot(self.ncs['v'],self.ncs['v_null_cline'],self.ncs['v'],self.ncs['w_null_cline'],self.eq_points['v_eq'],self.eq_points['w_eq'],'go')
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
		self.eq_points = {'v_eq':self.ncs['v'][idx], 'w_eq':self.ncs['w_null_cline'][idx]}

	def get_dvdt(self, v, w, i_ext=0):
		return (i_ext - (self.params['gca']*self.m_inf(v)*(v-self.params['eca'])) - (self.params['gk']*w*(v-self.params['ek'])) - (self.params['gl']*(v-self.params['el']))) / self.params['C']

	def get_dwdt(self, v, w):
		return (self.params['phi']*(self.w_inf(v)-w)) / self.tau_w(v)

	def simulate(self, v_init = None, w_init = None, i_ext = 0, timestep = 1e-3, end_time = None):
		self.stable = False
		if v_init is None:
			v_init = self.eq_points['v_eq']
		if w_init is None:
			w_init = self.eq_points['w_eq']
		if isinstance(i_ext, currents.CInput):
			i_ext.reset()	
		if end_time is not None:
			total_steps = int(end_time/timestep)
			v_s = np.zeros(total_steps+1)
			w_s = np.zeros(total_steps+1)
			i_s = np.zeros([total_steps+1,2])
			t_s = np.linspace(0, end_time, num=total_steps+1)
			v_s[0] = v_init
			w_s[0] = w_init
			i_s[0,0] = 0
			i_s[0,1] = 0
			if isinstance(i_ext, currents.CInput):
				for i in range(total_steps):
					i_now = i_ext.i_next()
					v_s[i+1], w_s[i+1] = self.simulate_step(v_s[i], w_s[i], timestep, i_now)
					i_s[i,0] = i_now[0]
					i_s[i,1] = i_now[1]

			else:
				for i in range(total_steps):
					v_s[i+1], w_s[i+1] = self.simulate_step(v_s[i], w_s[i], timestep, [i_ext,0])
					i_s[i,0] = i_ext
		else:
			max_limit = 1000
			min_limit = int(10 / timestep)
			eq_threshold = 1e-9
			v_s = [float(v_init)]
			w_s = [float(w_init)]
			i_s = [[0,0]]
			t_s = [0]
			t = 0
			v_int = float(v_init)
			w_int = float(w_init)
			if isinstance(i_ext, currents.CInput):
				while t < max_limit:
					if self.stable and i_ext.is_end():
						for i in range(min_limit):
							i_now = i_ext.i_next()
							i_s.append(i_now)
							v_int, w_int = self.simulate_step(v_int, w_int, timestep, i_now, eq_threshold)
							v_s.append(v_int)
							w_s.append(w_int)
							t += timestep
							t_s.append(t)
						break
					i_now = i_ext.i_next()
					i_s.append(i_now)
					v_int, w_int = self.simulate_step(v_int, w_int, timestep, i_now, eq_threshold)
					v_s.append(v_int)
					w_s.append(w_int)
					t += timestep
					t_s.append(t)	

			else:
				while t < max_limit:
					if self.stable:
						for i in range(min_limit):
							i_s.append([i_ext,0])
							v_int, w_int = self.simulate_step(v_int, w_int, timestep, [i_ext,0], eq_threshold)
							v_s.append(v_int)
							w_s.append(w_int)
							t += timestep
							t_s.append(t)
						break
					i_s.append([i_ext,0])
					v_int, w_int = self.simulate_step(v_int, w_int, timestep, [i_ext,0], eq_threshold)
					v_s.append(v_int)
					w_s.append(w_int)
					t += timestep
					t_s.append(t)

			v_s = np.asarray(v_s)
			w_s = np.asarray(w_s)
			i_s = np.asarray(i_s)
			t_s = np.asarray(t_s)		

		return {'Voltage':v_s, 'w':w_s, 'Timepoints':t_s, 'Currents':i_s}		

	def simulate_step(self, v, w, timestep, i_ext = [0,0], eq_threshold = 1e-9):
		dv = self.get_dvdt(v, w, i_ext[0])
		dw = self.get_dwdt(v, w)
		ds = np.abs(dv) + np.abs(dw*100)
		if ds < eq_threshold:
			self.stable = True
		v += (timestep * dv) + (i_ext[1] / self.params['C'])
		w += timestep * dw
		return v, w