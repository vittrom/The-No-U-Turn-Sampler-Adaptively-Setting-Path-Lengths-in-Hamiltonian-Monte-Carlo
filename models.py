import numpy as np
from urllib.request import urlopen

class mvn():
    def __init__(self, dim=100, off_diag=0.99):
        self.dim = dim
        self.off_diag=off_diag
        self.sigma = self.set_sigma()

    def set_sigma(self):
        sigma = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                sigma[i, j] = self.off_diag ** (np.abs(i-j))
        return sigma

    def U(self, q):
        return 0.5 * np.dot(np.dot(q, np.linalg.inv(self.sigma)), q)

    def grad_U(self, q):
        return np.array(np.dot(q, np.linalg.inv(self.sigma))).reshape(-1,)

    def params(self):
        return self.U, self.grad_U

class logReg():

    def __init__(self):
        self.X, self.y = self.set_data()
        self.normalize_data()

    def set_data(self):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
        raw_data = urlopen(url)
        credit = np.genfromtxt(raw_data)
        return credit[:, :-1], credit[:, -1:].squeeze()

    def normalize_data(self):
        mean_x = np.mean(self.X, axis=0)
        std_x = np.std(self.X, axis=0)
        self.X = (self.X - mean_x)/std_x
        self.y = self.y - 1

    def U(self, theta):
        X_theta = np.matmul(self.X, theta)
        exp_X_theta  = np.exp(X_theta)
        return -np.sum(self.y * X_theta - np.log(1 + exp_X_theta))

    def grad_U(self, theta):
        exp_X_theta = np.exp(np.matmul(self.X, theta))
        for i in range(exp_X_theta.shape[0]):
            if exp_X_theta[i] < 1e-10:
                exp_X_theta[i] = 0.0001
            if exp_X_theta[i] > 999999:
                exp_X_theta[i] = 999999
        div = self.y - exp_X_theta/(1 + exp_X_theta)
        grad = np.sum(np.multiply(div.reshape(-1, 1), self.X), axis=0)
        return -grad

    def params(self):
        return self.U, self.grad_U

class stochVolatility():

    def __init__(self, obs=1000, phi=0.98, kappa=0.65, sigma=0.15):
        self.phi_0 = phi
        self.kappa_0 = kappa
        self.sigma_0 = sigma
        self.obs = obs
        self.y = self.simulate_data()

    def simulate_data(self):
        T = self.obs
        count = 0
        y = np.zeros((T, 1))
        for i in range(T):
            if count == 0:
                x = np.random.normal(loc=0, scale=(self.sigma_0**2)/(1-self.phi_0**2), size=1)
                eps = np.random.normal(size = 1)
                y[count] = eps * self.kappa_0 * np.exp(0.5 * x)
            else:
                eta = np.random.normal(loc=0, scale=self.sigma_0**2)
                x = self.phi_0 * x + eta
                eps = np.random.normal(size=1)
                y[count] = eps * self.kappa_0 * np.exp(0.5 * x)

            count += 1
        return y

    def U(self, pars):
        x_s = pars[0:self.obs]
        theta = pars[self.obs:]
        alpha = theta[0]
        beta = theta[1]
        gamma = theta[2]

        log_p = self.obs * beta + np.sum(x_s/2) + np.sum((self.y ** 2)/(2 * np.exp(2*beta)*np.exp(x_s))) - \
                20.5*alpha + 22.5 * np.log(np.exp(alpha) + 1) + gamma * (self.obs/2 + 5) + \
                (2 * x_s[0]**2 * np.exp(alpha))/((np.exp(alpha) + 1)**2 * np.exp(1) * gamma) + \
                0.5 * np.sum(np.exp(-gamma) *(x_s[1:] - x_s[:-1] *(np.exp(alpha)- 1)/(np.exp(alpha)+ 1))**2)  + \
                0.25/np.exp(gamma)

        return log_p

    def grad_U(self, pars):
        x_s = pars[0:self.obs]
        theta = pars[self.obs:]
        alpha = theta[0]
        beta  = theta[1]
        gamma = theta[2]

        grad = np.zeros(self.obs + 3)
        grad[0] = 0.5 - (self.y[0]**2/(2 * np.exp(2*beta) * np.exp(x_s[0]))) + ((4 * x_s[0] * np.exp(alpha))/((np.exp(alpha) + 1)**2 * np.exp(1) * gamma)) - \
                  np.exp(-gamma) * (x_s[1] - ((np.exp(alpha) - 1)/(np.exp(alpha) + 1))*x_s[0]) * (np.exp(alpha) - 1)/(np.exp(alpha) + 1)

        for i in range(1, self.obs - 1):
            grad[i] = 0.5 - (self.y[i]**2/(2 * np.exp(2*beta) * np.exp(x_s[i]))) - \
                      np.exp(-gamma) * (x_s[i + 1] - x_s[i] * (np.exp(alpha) - 1)/(np.exp(alpha) + 1)) * (np.exp(alpha) - 1)/(np.exp(alpha) + 1) + \
                      np.exp(-gamma) * (x_s[i] - x_s[i - 1] * (np.exp(alpha) - 1) / (np.exp(alpha) + 1))

        i = self.obs - 1
        grad[i] = 0.5 - (self.y[i]**2/(2 * np.exp(2*beta) * np.exp(x_s[i]))) + \
                      np.exp(-gamma) * (x_s[i] - x_s[i - 1] * (np.exp(alpha) - 1) / (np.exp(alpha) + 1))

        #Extra 3 pars grad
        #grad_alpha
        i += 1
        grad[i] = -20.5 + 22.5 * (np.exp(alpha)/(1 + np.exp(alpha))) - \
                  (2 * x_s[0]**2 * np.exp(alpha) * (np.exp(alpha) - 1))/(np.exp(1) * gamma * (1 + np.exp(alpha))**3) - \
                  2 * np.exp(-gamma) * (np.exp(alpha) / (np.exp(alpha) + 1)**2) * np.sum(x_s[:-1] * (x_s[1:] - x_s[:-1] *((np.exp(alpha) - 1)/(np.exp(alpha)+ 1))))

        #grad beta
        i += 1
        grad[i] = self.obs - (np.exp(-2*beta) *np.sum(self.y**2/np.exp(x_s)))

        #grad gamma
        i += 1
        grad[i] = (self.obs/2 + 5) - (2 * x_s[0]**2 * np.exp(alpha))/(((np.exp(alpha) + 1) ** 2)*(gamma**2)) - \
                  0.5 * np.sum((x_s[1:] - x_s[:-1] * (np.exp(alpha) -1)/(np.exp(alpha) + 1))**2) * np.exp(-gamma) -\
                  0.25 * np.exp(-gamma)

        return grad

    def params(self):
        return self.U, self.grad_U

#Add sinusoidal sampling to test robustness to varying threshold