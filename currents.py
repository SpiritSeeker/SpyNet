import numpy as np

class CStep:
	"""docstring for CStep"""
	def __init__(self, mag, start_time = 0, timestep = 1e-3):
		self.mag = mag
		self.start_time = int(start_time / timestep)
		self.t = -1
		self.end = False

	def i(self):
		self.t += 1
		if self.t > self.start_time + 1000:
			self.end = True
		if self.t < self.start_time:
			return 0
		else:
			return self.mag

	def reset(self):
		self.t = -1
		self.end = False

class CPulse:
	"""docstring for CPulse"""
	def __init__(self, mag, width, start_time = 0, timestep = 1e-3):
		self.mag = mag
		self.start_time = int(start_time / timestep)
		self.width = int(width / timestep)
		self.t = -1
		self.end = False

	def i(self):
		self.t += 1
		if self.t > self.start_time + self.width + 1000:
			self.end = True
		if self.t < self.start_time:
			return 0
		elif self.t < (self.start_time + self.width):
			return self.mag
		else:
			return 0

	def reset(self):
		self.t = -1
		self.end = False	

class CImpulse:
	"""docstring for CImpulse"""
	def __init__(self, mag, start_time = 0, timestep = 1e-3):
		self.mag = mag
		self.start_time = int(start_time / timestep)
		self.t = -1
		self.end = False

	def i(self):
		self.t += 1
		if self.t > self.start_time + 1000:
			self.end = True
		if self.t == self.start_time:
			return self.mag
		else:
			return 0

	def reset(self):
		self.t = -1
		self.end = False	

class CInput:
	"""docstring for CInput"""
	def __init__(self):
		self.funcs = [CStep(0)]

	def add(self, f):
		self.funcs.append(f)

	def i_next(self):
		val = [0,0]
		for f in self.funcs:
			if isinstance(f, CImpulse):
				val[1] += f.i()
			else:
				val[0] += f.i()
		return val

	def reset(self):
		for i in self.funcs:
			i.reset()

	def clear(self):
		self.funcs = [CStep(0)]

	def is_end(self):
		q = True
		for i in self.funcs:
			if not i.end:
				q = False
		return q	