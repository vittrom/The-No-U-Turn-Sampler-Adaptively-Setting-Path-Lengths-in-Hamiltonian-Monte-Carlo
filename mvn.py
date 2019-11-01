import numpy as np

class mvn():
    def __init__(self, dim=100):
        self.dim = dim
        self.sigma = self.set_sigma()

    def set_sigma(self):
        sigma = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                sigma[i, j] = 0.99 ** (np.abs(i-j))
        return sigma

    def U(self, q):
        return 0.5 * np.dot(np.dot(q, np.linalg.inv(self.sigma)), q)

    def grad_U(self, q):
        return np.array(np.dot(q, np.linalg.inv(self.sigma))).reshape(-1,)

    def params(self):
        return self.U, self.grad_U