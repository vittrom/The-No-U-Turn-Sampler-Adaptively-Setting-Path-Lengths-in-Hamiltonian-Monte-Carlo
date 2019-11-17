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
        self.X = np.concatenate((np.repeat(1,  self.X.shape[0]).reshape(-1, 1), (self.X - mean_x)/std_x), axis=1)
        self.y = self.y - 1
        self.y[self.y == 0] = -1

    def U(self, theta):
        X_theta = - self.y * np.matmul(self.X, theta)
        idcs_neg = X_theta < -30
        idcs_pos = X_theta > 30
        idcs_rest = np.array([not j[0] and not j[1] for j in zip(idcs_pos, idcs_neg)])
        X_theta_1 = X_theta
        X_theta_1[idcs_rest] = -np.log1p(np.exp(X_theta_1[idcs_rest]))
        X_theta_1[idcs_neg] = -X_theta_1[idcs_neg]
        X_theta_1[idcs_pos] = -X_theta_1[idcs_pos]
        return np.sum(X_theta_1)

    def grad_U(self, theta):
        X_theta = -self.y * np.matmul(self.X, theta)
        idcs_neg = X_theta < -30
        idcs_pos = X_theta > 30
        idcs_rest = np.array([not j[0] and not j[1] for j in zip(idcs_pos, idcs_neg)])
        X_theta[idcs_pos] = 1
        X_theta[idcs_neg] = 1
        X_theta[idcs_rest] = np.exp(X_theta[idcs_rest])/(1 + np.exp(X_theta[idcs_rest]))
        grad = np.sum(X_theta[:, np.newaxis] * (self.y[:, np.newaxis] * self.X), axis=0)
        return grad

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
                eps = np.random.normal(size=1)
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
        if np.abs(alpha) < 30:
            phi = (np.exp(alpha) - 1) / (np.exp(alpha) + 1)
        elif alpha > 30:
            phi = 1
        elif alpha < -30:
            phi = -1
        beta = theta[1]
        if beta > 15:
            beta = 15
        elif beta < -15:
            beta = -15
        gamma = theta[2]
        if gamma > 30:
            gamma = 30
        elif gamma < -30:
            gamma = -30

        log_p = self.obs * beta + np.sum(x_s/2) + np.sum((self.y ** 2)/(2 * np.exp(2*beta)*np.exp(x_s))) - \
                20.5*alpha + 22.5 * np.log(2/(1-phi + 0.0001)) + gamma * (self.obs/2 + 5) + \
                (1-phi**2)*(2 * x_s[0]**2)/(4 * np.exp(gamma)) + \
                0.5 * np.sum(np.exp(-gamma) *(x_s[1:] - x_s[:-1] * phi)**2) + \
                0.25/np.exp(gamma)

        return log_p

    def grad_U(self, pars):
        x_s = pars[0:self.obs]
        theta = pars[self.obs:]
        alpha = theta[0]
        if alpha > 30:
            phi = 1
        elif alpha < -30:
            phi = -1
        else:
            phi = (np.exp(alpha) - 1) / (np.exp(alpha) + 1)
        beta = theta[1]
        if beta > 15:
            beta = 15
        elif beta < -15:
            beta = -15
        gamma = theta[2]
        if gamma > 30:
            gamma = 30
        elif gamma < -30:
            gamma = -30

        grad = np.zeros(self.obs + 3)
        grad[0] = 0.5 - (self.y[0]**2/(2 * np.exp(2*beta) * np.exp(x_s[0]))) + (x_s[0] * (1 - phi**2))/(np.exp(gamma)) - \
                  np.exp(-gamma) * (x_s[1] - phi*x_s[0]) * phi

        for i in range(1, self.obs - 1):
            grad[i] = 0.5 - (self.y[i]**2/(2 * np.exp(2*beta) * np.exp(x_s[i]))) - \
                      np.exp(-gamma) * (x_s[i + 1] - x_s[i] * phi) * phi + \
                      np.exp(-gamma) * (x_s[i] - x_s[i - 1] * phi)

        i = self.obs - 1
        grad[i] = 0.5 - (self.y[i]**2/(2 * np.exp(2*beta) * np.exp(x_s[i]))) + \
                      np.exp(-gamma) * (x_s[i] - x_s[i - 1] * phi)

        #Extra 3 pars grad
        #grad_alpha
        i += 1
        grad[i] = -20.5 + 22.5 * (1 + phi)/2 - \
                  (0.5 * x_s[0]**2 * (phi * (1-phi**2)))/(np.exp(gamma)) - \
                  0.5 * np.exp(-gamma) * (1-phi**2) / np.sum(x_s[:-1] * (x_s[1:] - x_s[:-1] *phi))

        #grad beta
        i += 1
        grad[i] = self.obs - (np.exp(-2*beta) *np.sum(self.y**2/np.exp(x_s)))

        #grad gamma
        i += 1
        grad[i] = (self.obs/2 + 5) - (0.5 * x_s[0]**2 * (1 - phi**2))/np.exp(gamma) - \
                  0.5 * np.sum((x_s[1:] - x_s[:-1] * phi)**2) * np.exp(-gamma) -\
                  0.25 * np.exp(-gamma)

        return grad

    def params(self):
        return self.U, self.grad_U

class banana():
    def __init__(self, B, dims):
        self.dims = dims
        self.B = B

    def U(self, theta):
        log_p = theta[0]**2/200 + 0.5 * (theta[1] + self.B * theta[0]**2 - 100 * self.B)**2
        if self.dims > 2:
            log_p += 0.5 * np.sum(theta[2:]**2)

        return log_p

    def grad_U(self, theta):
        grad = np.zeros(self.dims)

        grad[0] = theta[0]/100 + (theta[1] + self.B * theta[0]**2 - 100 * self.B) * 2 * self.B * theta[0]
        grad[1] = theta[1] + self.B * theta[0]**2 - 100 * self.B
        if self.dims > 2:
            grad[2:] = theta[2:]

        return grad

    def params(self):
        return self.U, self.grad_U