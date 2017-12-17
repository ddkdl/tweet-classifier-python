from Kernel import Kernel
import numpy as np

class PolynomialKernel(Kernel):
    def __init__(self, p=3):
        self.p = p

    def eval(self, X, Y):
        return (1 + np.dot(X, Y)) ** self.p