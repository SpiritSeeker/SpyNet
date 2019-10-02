import numpy as np
import matplotlib.pyplot as plt
import currents
from copy import deepcopy

# Add multiple eq points with J

class MLE_Neuron(object):
	"""docstring for MLE_Neuron"""
	def __init__(self, params = {'gca':4.4,'gk':8,'gl':2,'eca':120,'ek':-84,'el':-60,'phi':0.02,'V1':-1.2,'V2':18,'V3':2,'V4':30,'V5':2,'V6':30,'C':20}):
		self.params = deepcopy(params)
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


class HH_Neuron(object):
	"""docstring for HH_Neuron"""
	def __init__(self, params = {'gk':36,'gna':120,'gl':0.3,'ek':-72,'ena':55,'el':-50,'C':1,'an':-0.01,'bn':50,'cn':10,'dn':-1,'pn':0.125,'qn':60,'rn':80,'am':-0.1,'bm':35,'cm':10,'dm':-1,'pm':4,'qm':60,'rm':18,'bh':30,'ch':10,'dh':1,'ph':0.07,'qh':60,'rh':20}):
		self.params = deepcopy(params)
		self.get_eq_points(0)
		self.stable = True
	
	def alpha_n(self, v):
		return (np.finfo(float).eps * (self.params['an'] / ((-1 * self.params['bn'] / self.params['cn']) * np.exp((-1 * self.params['bn'] / self.params['cn'])))) + self.params['an'] * (v + self.params['bn'])) / (np.finfo(float).eps + np.exp(-1 * (v + self.params['bn']) / self.params['cn']) + self.params['dn'])

	def beta_n(self, v):
		return self.params['pn'] * np.exp(-1 * (v + self.params['qn']) / self.params['rn'])

	def alpha_m(self, v):
		return (np.finfo(float).eps * (self.params['an'] / ((-1 * self.params['bn'] / self.params['cn']) * np.exp((-1 * self.params['bn'] / self.params['cn'])))) + self.params['am'] * (v + self.params['bm'])) / (np.finfo(float).eps + np.exp(-1 * (v + self.params['bm']) / self.params['cm']) + self.params['dm'])

	def beta_m(self, v):
		return self.params['pm'] * np.exp(-1 * (v + self.params['qm']) / self.params['rm'])	

	def alpha_h(self, v):
		return self.params['ph'] * np.exp(-1 * (v + self.params['qh']) / self.params['rh'])

	def beta_h(self, v):
		return 1 / (np.exp(-1 * (v + self.params['bh']) / self.params['ch']) + self.params['dh'])

	def n_inf(self, v):
		return self.alpha_n(v) / (self.alpha_n(v) + self.beta_n(v))

	def m_inf(self, v):
		return self.alpha_m(v) / (self.alpha_m(v) + self.beta_m(v))

	def h_inf(self, v):
		return self.alpha_h(v) / (self.alpha_h(v) + self.beta_h(v))	

	def get_dvdt(self, v, n, m, h, i_ext):
		return (i_ext - (self.params['gk'] * (n**4) * (v - self.params['ek'])) - (self.params['gna'] * (m**3) * h * (v - self.params['ena'])) - (self.params['gl'] * (v - self.params['el']))) / self.params['C']

	def get_dndt(self, v, n):
		return (self.alpha_n(v) * (1-n)) - (self.beta_n(v) * n)

	def get_dmdt(self, v, m):
		return (self.alpha_m(v) * (1-m)) - (self.beta_m(v) * m)

	def get_dhdt(self, v, h):
		return (self.alpha_h(v) * (1-h)) - (self.beta_h(v) * h)

	def get_eq_points(self, i_ext):
		v_lower = min(self.params['ek'],self.params['el'],self.params['ena'])
		v_upper = max(self.params['ek'],self.params['el'],self.params['ena'])
		v = np.linspace(v_lower, v_upper, (v_upper-v_lower)/0.0001, endpoint=True)
		v_nc = (i_ext - (self.params['gk']*(self.n_inf(v)**4)*(v-self.params['ek']))-(self.params['gl']*(v-self.params['el']))) / (np.finfo(float).eps+self.params['gna']*(self.m_inf(v)**3)*(v-self.params['ena']))
		h_nc = self.h_inf(v)
		# self.ncs = {'v':v, 'v_null_cline':v_nc, 'h_null_cline':h_nc}
		idx = np.argwhere(np.diff(np.sign(v_nc - h_nc))).flatten()
		v_eq = np.min(v[idx])
		self.eq_points = {'v_eq':v_eq, 'n_eq':self.n_inf(v_eq), 'm_eq':self.m_inf(v_eq), 'h_eq':self.h_inf(v_eq)}

	def simulate(self, v_init = None, n_init = None, m_init = None, h_init = None, i_ext = 0, timestep = 1e-3, end_time = None):
		self.stable = False
		if v_init is None:
			v_init = self.eq_points['v_eq']
		if n_init is None:
			n_init = self.eq_points['n_eq']
		if m_init is None:
			m_init = self.eq_points['m_eq']
		if h_init is None:
			h_init = self.eq_points['h_eq']
		if isinstance(i_ext, currents.CInput):
			i_ext.reset()
		elif isinstance(i_ext, int) or isinstance(i_ext, float):
			val = i_ext
			i_ext = currents.CInput()
			i_ext.add(currents.CStep(val, timestep = timestep))

		if end_time is not None:
			total_steps = int(end_time/timestep)
			v_s = np.zeros(total_steps+1)
			n_s = np.zeros(total_steps+1)
			m_s = np.zeros(total_steps+1)
			h_s = np.zeros(total_steps+1)
			i_s = np.zeros([total_steps+1,2])
			t_s = np.linspace(0, end_time, num = total_steps + 1)
			v_s[0] = v_init
			n_s[0] = n_init
			m_s[0] = m_init
			h_s[0] = h_init
			i_s[0,0] = 0
			i_s[0,1] = 0
			for i in range(total_steps):
				i_now = i_ext.i_next()
				v_s[i+1], n_s[i+1], m_s[i+1], h_s[i+1] = self.simulate_step(v_s[i], n_s[i], m_s[i], h_s[i], timestep, i_now)
				i_s[i+1,0] = i_now[0]
				i_s[i+1,1] = i_now[1]

		else:
			max_limit = 1000
			min_limit = int(10 / timestep)
			v_s = [float(v_init)]
			n_s = [float(n_init)]
			m_s = [float(m_init)]
			h_s = [float(h_init)]
			i_s = [[0,0]]		
			t_s = [0]
			t = 0
			v_int = float(v_init)
			n_int = float(n_init)
			m_int = float(m_init)
			h_int = float(h_init)
			while t < max_limit:
				if self.stable and i_ext.is_end():
					for i in range(min_limit):
						i_now = i_ext.i_next()
						i_s.append(i_now)
						v_int, n_int, m_int, h_int = self.simulate_step(v_int, n_int, m_int, h_int, timestep, i_now)
						v_s.append(v_int)
						n_s.append(n_int)
						m_s.append(m_int)
						h_s.append(h_int)
						t += timestep
						t_s.append(t)
					break
				i_now = i_ext.i_next()
				i_s.append(i_now)
				v_int, n_int, m_int, h_int = self.simulate_step(v_int, n_int, m_int, h_int, timestep, i_now)
				v_s.append(v_int)
				n_s.append(n_int)
				m_s.append(m_int)
				h_s.append(h_int)
				t += timestep
				t_s.append(t)
			v_s = np.asarray(v_s)
			n_s = np.asarray(n_s)
			m_s = np.asarray(m_s)
			h_s = np.asarray(h_s)

		return {'Voltage':v_s, 'n':n_s, 'm':m_s, 'h':h_s, 'Timepoints':t_s, 'Currents':i_s}		


	def simulate_step(self, v, n, m, h, timestep, i_ext, eq_threshold = 1e-8):
		dv = self.get_dvdt(v, n, m, h, i_ext[0])
		dn = self.get_dndt(v, n)
		dm = self.get_dmdt(v, m)
		dh = self.get_dhdt(v, h)
		ds = np.abs(dv) + np.abs(dn*100) + np.abs(dm*100) + np.abs(dh*100)
		if ds < eq_threshold:
			self.stable = True
		v += (timestep * dv) + (i_ext[1] / self.params['C'])
		n += timestep * dn
		m += timestep * dm
		h += timestep * dh
		return v, n, m, h