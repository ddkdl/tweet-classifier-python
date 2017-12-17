import numpy as np
from Kernel import Kernel

class LinearKernel(Kernel):
    def __init__(self):
        pass

    def eval(self, X, Y):
        return np.dot(X, Y)