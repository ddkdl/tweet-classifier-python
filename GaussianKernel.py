from Kernel import Kernel
import numpy as np

class GaussianKernel(Kernel):
    def __init__(self, sigma=5.0):
        self.sigma = sigma

    def eval(self, X, Y):
        return np.exp(-np.linalg.norm(X-Y)**2 / (2 * (self.sigma ** 2)))