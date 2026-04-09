"""
@User: sandruskyi
"""
import numpy as np
from scipy import stats

class StatisticsMeter:
    def __init__(self):
        self.values = []

    def update(self, value, size):
        self.values.append(value)

    def mean(self):
        return np.mean(self.values)

    def std(self):
        return np.std(self.values)

    def mode(self):
        return stats.mode(self.values, axis=None, keepdims=None)[0] # Return only the mode
