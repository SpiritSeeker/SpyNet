class CStep:
    """
    Class for representing a step input.
    """
    def __init__(self, mag, start_time=0, timestep=1e-3):
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
        return self.mag

    def reset(self):
        self.t = -1
        self.end = False

class CPulse:
    """docstring for CPulse"""
    def __init__(self, mag, width, start_time=0, timestep=1e-3):
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
        if self.t < (self.start_time + self.width):
            return self.mag
        return 0

    def reset(self):
        self.t = -1
        self.end = False

class CImpulse:
    """docstring for CImpulse"""
    def __init__(self, mag, start_time=0, timestep=1e-3):
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
        return 0

    def reset(self):
        self.t = -1
        self.end = False

class CInput:
    """
    Class for holding all functions and generating values when called.
    """
    def __init__(self):
        """
        Constructor for the class CInput. Initializes to a step with 0 magnitude.
        """
        self.funcs = [CStep(0)]

    def add(self, f):
        """
        Adds a function to the class. f is an instance of functions defined in currents.
        """
        self.funcs.append(f)

    def i_next(self):
        """
        Returns the value at next timestep.
        """
        val = [0, 0]
        for f in self.funcs:
            if isinstance(f, CImpulse):
                val[1] += f.i()
            else:
                val[0] += f.i()
        return val

    def reset(self):
        """
        Sets time to zero.
        """
        for i in self.funcs:
            i.reset()

    def clear(self):
        """
        Reinitializes to a step with 0 magnitude.
        """
        self.funcs = [CStep(0)]

    def is_end(self):
        """
        Returns True if there is no change in the functional value in the future, False otherwise.
        Can be used in combination with system stability for early stopping.
        """
        q = True
        for i in self.funcs:
            if not i.end:
                q = False
        return q
