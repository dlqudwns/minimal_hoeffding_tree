import copy

import numpy as np

class Mean:

    def __init__(self):
        self.n = 0
        self._mean = 0.0

    def update(self, x, w=1.0):
        self.n += w
        self._mean += (w / self.n) * (x - self._mean)
        return self

    def get(self):
        return self._mean

    @classmethod
    def _from_state(cls, n, mean):
        new = cls()
        new.n = n
        new._mean = mean
        return new

    def __iadd__(self, other):
        old_n = self.n
        self.n += other.n
        self._mean = (old_n * self._mean + other.n * other.get()) / self.n
        return self

    def __add__(self, other):
        result = copy.deepcopy(self)
        result += other
        return result

    def __isub__(self, other):
        old_n = self.n
        self.n -= other.n

        if self.n > 0:
            self._mean = (old_n * self._mean - other.n * other._mean) / self.n
        else:
            self.n = 0.0
            self._mean = 0.0
        return self

    def __sub__(self, other):
        result = copy.deepcopy(self)
        result -= other
        return result

