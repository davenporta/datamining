import numpy as np

class LinearRegression:
    """Linear regression with numpy arrays"""
    def __init__ (self, A, b):
        self.A = A
        self.b = b

    def fit(self):
        """returns tuple (m,b)"""
        self.sigma = np.linalg.inv(self.A.T.dot(self.A)).dot(self.A.T.dot(self.b))
        return (self.sigma[0], self.sigma[1])
