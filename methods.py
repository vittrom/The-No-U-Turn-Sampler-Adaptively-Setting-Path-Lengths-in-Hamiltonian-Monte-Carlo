from utils import *
from time import process_time

class HMC():

    def __init__(self, U, grad_U, start_position, extra_pars, iter, samp_random=True, dual_averaging=False):
        self.U = U
        self.grad_U = grad_U
        self.current_position = start_position
        self.dim = len(start_position)
        self.iter = iter
        self.samp_random = samp_random
        self.grad_evals = 0

        self.L_noise = self.epsilon_noise = self.epsilon_iter = self.L_center = None
        self.set_extra_pars(extra_pars)

        self.elapsed_time = 0

        self.epsilon = None
        self.L = None
        self.alpha = None
        self.curr_iter = 1
        self.dual_averaging = dual_averaging
        self.adapt_epsilon = 0
        if self.dual_averaging:
            self.h_bar = 0
            self.mu = self.epsilon_bar = self.gamma = self.t0 = self.kappa = self.delta = None
            self.set_dual_averaging_pars(extra_pars)

    def set_dual_averaging_pars(self, extra_pars):
        self.epsilon_bar = extra_pars["epsilon_bar"]
        self.gamma = extra_pars["gamma"]
        self.t0 = extra_pars["t0"]
        self.kappa = extra_pars["kappa"]
        self.delta = extra_pars["delta"]
        self.adapt_epsilon = extra_pars["adapt_epsilon"]
        if 0 <= extra_pars["adapt_epsilon"] <= 1:
            self.adapt_epsilon = np.int(np.round(self.adapt_epsilon * self.iter))

    def set_extra_pars(self, extra_pars):
        self.L_noise = extra_pars["L_noise"]
        self.epsilon_noise = extra_pars["epsilon_noise"]
        self.epsilon_iter = extra_pars["epsilon_start"]
        self.L_center = extra_pars["L_start"]

    def leapfrog(self, position, momentum):
        momentum -= self.epsilon * self.grad_U(position) / 2
        position += self.epsilon * momentum
        momentum -= self.epsilon * self.grad_U(position) / 2
        return [position, momentum]

    def step(self):
        momentum = np.random.normal(size=self.dim)
        position = deepcopy(self.current_position)
        current_momentum = deepcopy(momentum)

        self.L = np.random.randint(low=np.int(np.floor(self.L_center * (1 - self.L_noise))), high=np.int(np.ceil(self.L_center * (1 + self.L_noise))))
        if self.dual_averaging and self.curr_iter <= self.adapt_epsilon:
                self.epsilon = self.epsilon_iter
        else:
            self.epsilon = np.random.uniform(low=self.epsilon_iter * (1 - self.epsilon_noise),
                                             high=self.epsilon_iter * (1 + self.epsilon_noise), size=1)

        if self.samp_random:
            positions = np.zeros((self.L, self.dim))
            momentums = np.zeros((self.L, self.dim))

        for i in range(self.L):
            [position,  momentum] = self.leapfrog(position, momentum)
            if self.samp_random:
                positions[i, :] = position
                momentums[i, :] = momentum

        self.grad_evals += self.L

        if self.samp_random:
            idx = np.random.randint(self.L)
            position = positions[idx, :]
            momentum = momentums[idx, :]

        momentum = - momentum
        acc_ratio = compute_acc_ratio(self.current_position, current_momentum, position, momentum, self.U)

        self.alpha = np.min((1, acc_ratio))
        if np.random.uniform(size=1) < self.alpha:
            self.current_position = position

        if self.dual_averaging:
            if self.curr_iter <= self.adapt_epsilon:
                self.adaptation()
            else:
                self.epsilon_iter = self.epsilon_bar

    def simulate(self):
        results = np.zeros((self.iter + self.adapt_epsilon, self.dim))

        t0 = process_time()
        if self.dual_averaging:
            curr_pos = deepcopy(self.current_position)
            self.epsilon_iter = find_reasonable_epsilon(curr_pos, self.grad_U, self.U)
            self.mu = np.log(10 * self.epsilon_iter)

        for i in range(self.iter + self.adapt_epsilon):
            self.step()
            results[i, :] = self.current_position
            self.curr_iter += 1
        t1 = process_time()
        self.elapsed_time = t1 - t0
        return results

    def adaptation(self):
        frac = 1/(self.curr_iter + self.t0)
        self.h_bar = (1 - frac) * self.h_bar + frac * (self.delta - self.alpha)
        self.epsilon_iter = np.array(np.exp(self.mu - (np.sqrt(self.curr_iter)/self.gamma) * self.h_bar)).reshape(-1,)
        eta = self.curr_iter ** -self.kappa
        self.epsilon_bar = np.exp(eta * np.log(self.epsilon_iter) + (1 - eta) * np.log(self.epsilon_bar))

class HMC_wiggle():

    def __init__(self, U, grad_U, start_position, extra_pars, iter, samp_random=True, dual_averaging=False):
        self.U = U
        self.grad_U = grad_U
        self.current_position = start_position
        self.dim = len(start_position)
        self.iter = iter
        self.samp_random = samp_random
        self.angles_sum = 0
        self.grad_evals = 0
        self.epsilon = None
        self.L = list()
        self.traj_length = list()

        self.L_noise = self.threshold = self.epsilon_noise = self.epsilon_iter = None
        self.set_extra_pars(extra_pars)

        self.elapsed_time = 0

        self.version = self.adapt_L = self.function = self.method = self.quantile_ub = self.quantile_lb = None
        self.set_version(extra_pars)

        self.alpha = None
        self.curr_iter = 1
        self.dual_averaging = dual_averaging
        self.adapt_epsilon = 0
        if self.dual_averaging:
            self.h_bar = 0
            self.mu = self.epsilon_bar = self.gamma = self.t0 = self.kappa = self.delta = None
            self.set_dual_averaging_pars(extra_pars)

    def set_dual_averaging_pars(self, extra_pars):
        self.epsilon_bar = extra_pars["epsilon_bar"]
        self.gamma = extra_pars["gamma"]
        self.t0 = extra_pars["t0"]
        self.kappa = extra_pars["kappa"]
        self.delta = extra_pars["delta"]
        self.adapt_epsilon = extra_pars["adapt_epsilon"]
        if 0 <= extra_pars["adapt_epsilon"] <= 1:
            self.adapt_epsilon = np.int(np.round(self.adapt_epsilon * self.iter))

    def set_version(self, extra_pars):
        try:
            self.version = extra_pars["version"]
        except:
            self.version = "vanilla"
        if self.version != "vanilla":
            self.adapt_L = extra_pars["adapt_L"]
            if 0 < self.adapt_L < 1:
                self.adapt_L = np.int(np.round(self.adapt_L * self.iter))

        if self.version == "vanish_vanilla":
            self.function = extra_pars["fn"]
        if self.version == "distr_L":
            self.method = extra_pars["method"]
            if self.method == "quantile":
                self.quantile_lb = extra_pars["quantile_lb"]
                self.quantile_ub = extra_pars["quantile_ub"]

    def set_extra_pars(self, extra_pars):
        self.L_noise = extra_pars["L_noise"]
        self.threshold = extra_pars["threshold"]
        self.epsilon_noise = extra_pars["epsilon_noise"]
        self.epsilon_iter = extra_pars["epsilon_start"]

    def leapfrog(self, position, momentum):
        start_position = deepcopy(position)

        momentum -= self.epsilon * self.grad_U(position) / 2
        position += self.epsilon * momentum
        momentum -= self.epsilon * self.grad_U(position) / 2

        diff_pos = position - start_position
        diff_pos = normalize_vec(diff_pos)
        norm_mom = normalize_vec(momentum)

        self.angles_sum += np.arccos(np.dot(diff_pos, norm_mom)) * 180 / np.pi
        return [position, momentum]

    def step(self):
        self.angles_sum = 0
        momentum = np.random.normal(size=self.dim)
        position = deepcopy(self.current_position)
        current_momentum = deepcopy(momentum)

        if self.dual_averaging and self.curr_iter <= self.adapt_epsilon:
                self.epsilon = self.epsilon_iter
        else:
            self.epsilon = np.random.uniform(low=self.epsilon_iter * (1 - self.epsilon_noise),
                                             high=self.epsilon_iter * (1 + self.epsilon_noise), size=1)

        if self.samp_random:
            positions = list()
            momentums = list()

        L = 0
        while self.angles_sum <= self.threshold:
            if L > 100:
                self.threshold *= 0.98
                break
            # print(stop_criterion(self.current_position, position, current_momentum, momentum))
            # if L > 0 and not stop_criterion(self.current_position, position, current_momentum, momentum):
            #     break
            L += 1
            [position,  momentum] = self.leapfrog(position, momentum)
            if self.samp_random:
                positions.append(position)
                momentums.append(momentum)

        if self.angles_sum > 1.05 * self.threshold:
            self.threshold = np.min((180, self.threshold * 1.05))

        self.grad_evals += L
        self.L.append(L)
        self.traj_length.append(L)

        if self.samp_random:
            idx = np.random.randint(L)
            position = positions[idx]
            momentum = momentums[idx]

        momentum = - momentum
        acc_ratio = compute_acc_ratio(self.current_position, current_momentum, position, momentum, self.U)

        self.alpha = np.min((1, acc_ratio))
        if np.random.uniform(size=1) < self.alpha:
            self.current_position = position

        if self.dual_averaging:
            if self.curr_iter <= self.adapt_epsilon:
                self.adaptation()
            else:
                self.epsilon_iter = self.epsilon_bar

    def simple_hmc_step(self, L):
        momentum = np.random.normal(size=self.dim)
        position = deepcopy(self.current_position)
        current_momentum = deepcopy(momentum)

        self.epsilon = np.random.uniform(low=self.epsilon_iter * (1 - self.epsilon_noise),
                                         high=self.epsilon_iter * (1 + self.epsilon_noise), size=1)
        if self.samp_random:
            positions = np.zeros((L, self.dim))

        for i in range(L):
            [position, momentum] = self.leapfrog(position, momentum)
            if self.samp_random:
                positions[i, :] = position

        if self.samp_random:
            position = positions[np.random.randint(L), :]

        momentum = - momentum
        current_U = self.U(self.current_position)
        current_K = np.sum(current_momentum ** 2) / 2
        proposed_U = self.U(position)
        proposed_K = np.sum(momentum ** 2) / 2

        self.alpha = np.min((1, np.exp(current_U - proposed_U + current_K - proposed_K)))
        if np.random.uniform(size=1) < self.alpha:
            self.current_position = position

    def simulate(self):
        results = np.zeros((self.iter + self.adapt_epsilon, self.dim))

        t0 = process_time()
        if self.dual_averaging:
            curr_pos = deepcopy(self.current_position)
            self.epsilon_iter = find_reasonable_epsilon(curr_pos, self.grad_U, self.U)
            self.mu = np.log(10 * self.epsilon_iter)

        if self.version == "vanilla":
            for i in range(self.iter + self.adapt_epsilon):
                self.step()
                results[i, :] = self.current_position
                self.curr_iter += 1
        if self.version == "vanish_vanilla" or self.version == "distr_L":
            for i in range(self.iter + self.adapt_epsilon):
                if self.curr_iter == self.adapt_epsilon - 1:
                    self.L = list()
                if self.curr_iter < self.adapt_epsilon + self.adapt_L:
                    self.step()
                    results[i, :] = self.current_position
                if self.curr_iter >= self.adapt_epsilon + self.adapt_L:
                    self.L = self.L[0:self.adapt_L]
                    if self.version == "vanish_vanilla":
                        L_center = self.function(self.L)
                        L = np.random.randint(low=np.int(np.floor(L_center * (1-self.L_noise))),
                                              high=np.int(np.ceil(L_center * (1 + self.L_noise))))
                    if self.version == "distr_L":
                        if self.method == "quantile":
                            bounds = np.quantile(self.L, [self.quantile_lb, self.quantile_ub], interpolation="nearest")
                            if bounds[0] == bounds[1]:
                                bounds[1] +=1
                            L = np.random.randint(low=bounds[0], high=bounds[1])
                        if self.method == "random":
                            L = np.random.choice(self.L)
                        if self.method == "random_unif":
                            L = np.random.choice(self.L)
                            if L == 1:
                                L += 1
                            L = np.random.randint(low=1, high=L, size=1)

                    self.simple_hmc_step(np.int(L))
                    results[i, :] = self.current_position

                self.curr_iter += 1
        t1 = process_time()
        self.elapsed_time = t1 - t0
        return results

    def adaptation(self):
        frac = 1/(self.curr_iter + self.t0)
        self.h_bar = (1 - frac) * self.h_bar + frac * (self.delta - self.alpha)
        self.epsilon_iter = np.array(np.exp(self.mu - (np.sqrt(self.curr_iter)/self.gamma) * self.h_bar)).reshape(-1,)
        eta = self.curr_iter ** -self.kappa
        self.epsilon_bar = np.exp(eta * np.log(self.epsilon_iter) + (1 - eta) * np.log(self.epsilon_bar))

class eHMC():

    def __init__(self, U, grad_U, start_position, extra_pars, iter, samp_random=True, dual_averaging=False):
        self.U = U
        self.grad_U = grad_U
        self.current_position = start_position
        self.dim = len(start_position)
        self.iter = iter
        self.samp_random = samp_random

        self.grad_evals = 0
        self.traj_length = list()
        self.elapsed_time = 0

        self.epsilon = None
        self.L = list()

        self.L_noise = self.threshold = self.epsilon_noise = self.epsilon_iter = None
        self.set_extra_pars(extra_pars)

        self.version = self.adapt_L = self.function = self.method = self.quantile_ub = self.quantile_lb = None
        self.set_version(extra_pars)

        self.alpha = None
        self.curr_iter = 1
        self.dual_averaging = dual_averaging

        if self.dual_averaging:
            self.h_bar = 0
            self.mu = self.epsilon_bar = self.gamma = self.t0 = self.kappa = self.delta = self.adapt_epsilon = None
            self.set_dual_averaging_pars(extra_pars)
        self.current_position = start_position
        self.U = U
        self.grad_U = grad_U
        self.dim = len(start_position)
        self.iter = iter
        self.alpha = None

        self.L0 = self.epsilon = self.epsilon_noise = self.epsilon_iter = self.adapt_L = self.adapt_with_opt_epsilon = None
        self.set_extra_pars(extra_pars)

        self.L = list()

        self.epsilon_bar = self.gamma = self.t0 = self.kappa = self.delta = self.adapt_epsilon = None
        self.set_dual_averaging_pars(extra_pars)

        self.version = self.quantile = None
        self.set_version(extra_pars)

    def set_version(self, extra_pars):
        try:
            self.version = extra_pars["version"]
        except:
            self.version = "vanilla"

        if self.version == "quantile":
            self.quantile = extra_pars["quantile_ub"]

    def set_extra_pars(self, extra_pars):
        self.L0 = extra_pars["L_start"]
        self.epsilon_iter = extra_pars["epsilon_start"]
        self.epsilon_noise = extra_pars["epsilon_noise"]
        self.adapt_L = extra_pars["adapt_L"]
        if 0 < self.adapt_L < 1:
            self.adapt_L = np.int(np.round(self.adapt_L * self.iter))
        self.adapt_with_opt_epsilon = extra_pars["adapt_with_opt_epsilon"]

    def set_dual_averaging_pars(self, extra_pars):
        self.epsilon_bar = extra_pars["epsilon_bar"]
        self.gamma = extra_pars["gamma"]
        self.t0 = extra_pars["t0"]
        self.kappa = extra_pars["kappa"]
        self.delta = extra_pars["delta"]
        self.adapt_epsilon = extra_pars["adapt_epsilon"]
        if 0 <= extra_pars["adapt_epsilon"] <= 1:
            self.adapt_epsilon = np.int(np.round(self.adapt_epsilon * self.iter))

    def leapfrog(self, position, momentum):
        momentum -= self.epsilon * self.grad_U(position) / 2
        position += self.epsilon * momentum
        momentum -= self.epsilon * self.grad_U(position) / 2

        return [position, momentum]

    def longest_batch(self, position):
        current_position = deepcopy(position)
        momentum = np.random.normal(size=self.dim)
        current_momentum = deepcopy(momentum)
        increment = stop = stop_ind = iter_batch = 0

        while increment >= 0 or iter_batch < self.L0:
            iter_batch += 1
            [position, momentum] = self.leapfrog(position, momentum)

            if stop == 0:
                position_diff = position - current_position
                increment = np.dot(position_diff, momentum)
                if np.isnan(increment):
                    return [current_position, iter_batch, True]
                if increment < 0:
                    stop_ind = iter_batch
                    stop = 1
            if iter_batch == self.L0 - 1:
                break
            if iter_batch > 1e4:
                return [current_position, iter_batch, True]

        momentum = - momentum
        current_U = self.U(current_position)
        current_K = np.sum(current_momentum ** 2) / 2
        proposed_U = self.U(position)
        proposed_K = np.sum(momentum ** 2) / 2

        alpha = current_U + current_K - proposed_U - proposed_K
        if np.isnan(alpha):
            return [current_position, None, True]
        else:
            if np.log(np.random.uniform(size=1)) < alpha:
                return [position, stop_ind, False]
            else:
                return [current_position, stop_ind, False]

    def learnL(self):
        self.L.append(self.L0)
        position = deepcopy(self.current_position)
        for i in range(self.adapt_L):
            L = np.quantile(self.L, 0.95, interpolation="nearest")
            L = 2 if L <= 1 else L
            self.L0 = np.random.randint(low=1, high=L)
            [position, L_emp, diverge] = self.longest_batch(position)
            if not diverge:
                self.L.append(L_emp)

    def adaptation(self):
        frac = 1 / (self.curr_iter + self.t0)
        self.h_bar = (1 - frac) * self.h_bar + frac * (self.delta - self.alpha)
        self.epsilon_iter = np.array(np.exp(self.mu - (np.sqrt(self.curr_iter) / self.gamma) * self.h_bar)).reshape(
            -1, )
        eta = self.curr_iter ** -self.kappa
        self.epsilon_bar = np.exp(eta * np.log(self.epsilon_iter) + (1 - eta) * np.log(self.epsilon_bar))

    def step(self):
        momentum = np.random.normal(size=self.dim)
        position = deepcopy(self.current_position)
        current_momentum = deepcopy(momentum)

        if self.dual_averaging and self.curr_iter <= self.adapt_epsilon:
            self.epsilon = self.epsilon_iter
        else:
            self.epsilon = np.random.uniform(low=self.epsilon_iter * (1 - self.epsilon_noise),
                                             high=self.epsilon_iter * (1 + self.epsilon_noise), size=1)

        if self.samp_random:
            positions = np.zeros((self.L0, self.dim))
            momentums = np.zeros((self.L0, self.dim))

        for i in range(self.L0):
            [position,  momentum] = self.leapfrog(position, momentum)
            if self.samp_random:
                positions[i, :] = position
                momentums[i, :] = momentum

        self.grad_evals += self.L0
        self.traj_length.append(self.L0)

        if self.samp_random:
            idx = np.random.randint(self.L0)
            position = positions[idx, :]
            momentum = momentums[idx, :]

        momentum = - momentum
        acc_ratio = compute_acc_ratio(self.current_position, current_momentum, position, momentum, self.U)

        self.alpha = np.min((1, acc_ratio))
        if np.random.uniform(size=1) < self.alpha:
            self.current_position = position

        if self.dual_averaging:
            if self.curr_iter <= self.adapt_epsilon:
                self.adaptation()
            else:
                self.epsilon_iter = self.epsilon_bar

    def simulate(self):
        results = np.zeros((self.iter + self.adapt_epsilon, self.dim))

        t0 = process_time()
        if self.adapt_with_opt_epsilon:
            start_pos = deepcopy(self.current_position)

        if self.dual_averaging:
            curr_pos = deepcopy(self.current_position)
            self.epsilon_iter = find_reasonable_epsilon(curr_pos, self.grad_U, self.U)
            self.mu = np.log(10 * self.epsilon_iter)
            self.epsilon = self.epsilon_iter

        if not self.adapt_with_opt_epsilon:
            self.learnL()

        for i in range(self.iter + self.adapt_epsilon):
            if i > self.adapt_epsilon:
                if self.version == "vanilla":
                    self.L0 = np.random.choice(self.L)
                    self.L0 = 2 if self.L0 <= 1 else self.L0
                if self.version == "quantile":
                    self.L0 = np.quantile(self.L, self.quantile, interpolation="nearest")
                if self.version == "uniform":
                    self.L0 = np.random.choice(self.L)
                    self.L0 = 2 if self.L0 <= 1 else self.L0
                    self.L0 = np.random.randint(low=1, high=self.L0)
            self.step()
            results[i, :] = self.current_position
            if self.adapt_with_opt_epsilon:
                if i == self.adapt_epsilon - 1:
                    self.learnL()
                    self.current_position = start_pos
            self.curr_iter += 1
        t1 = process_time()
        self.elapsed_time = t1 - t0
        return results

class prHMC():

    def __init__(self, U, grad_U, start_position, extra_pars, iter, dual_averaging=False):
        self.U = U
        self.grad_U = grad_U
        self.current_position = start_position
        self.current_momentum = None
        self.dim = len(start_position)
        self.iter = iter
        self.angles_sum = 0
        self.w_plus = list()
        self.epsilon = None
        self.L = list()

        self.grad_evals = 0
        self.traj_length = list()

        self.L_noise = self.threshold = self.epsilon_noise = self.epsilon_iter = self.refreshment_prob = None
        self.set_extra_pars(extra_pars)

        self.elapsed_time = 0

        self.version = self.adapt_L = self.function = self.method = self.quantile_ub = self.quantile_lb = None
        self.set_version(extra_pars)

        self.alpha = None
        self.curr_iter = 1
        self.dual_averaging = dual_averaging

        if self.dual_averaging:
            self.h_bar = 0
            self.mu = self.epsilon_bar = self.gamma = self.t0 = self.kappa = self.delta = self.adapt_epsilon = None
            self.set_dual_averaging_pars(extra_pars)
        self.current_position = start_position
        self.U = U
        self.grad_U = grad_U
        self.dim = len(start_position)
        self.iter = iter
        self.alpha = None

        self.L0 = self.epsilon = self.epsilon_noise = self.epsilon_iter = self.adapt_L = self.adapt_with_opt_epsilon = None
        self.set_extra_pars(extra_pars)

        self.L = list()

        self.epsilon_bar = self.gamma = self.t0 = self.kappa = self.delta = self.adapt_epsilon = None
        self.set_dual_averaging_pars(extra_pars)

        self.version = self.quantile = None
        self.set_version(extra_pars)

        self.direction = self.l = self.current_i = None

    def set_version(self, extra_pars):
        try:
            self.version = extra_pars["version"]
        except:
            self.version = "vanilla"

        if self.version == "quantile":
            self.quantile = extra_pars["quantile_ub"]

    def set_extra_pars(self, extra_pars):
        self.L0 = extra_pars["L_start"]
        self.epsilon_iter = extra_pars["epsilon_start"]
        self.epsilon_noise = extra_pars["epsilon_noise"]
        self.adapt_L = extra_pars["adapt_L"]
        if 0 < self.adapt_L < 1:
            self.adapt_L = np.int(np.round(self.adapt_L * self.iter))
        self.adapt_with_opt_epsilon = extra_pars["adapt_with_opt_epsilon"]
        self.refreshment_prob = extra_pars["refreshment_prob"]

    def set_dual_averaging_pars(self, extra_pars):
        self.epsilon_bar = extra_pars["epsilon_bar"]
        self.gamma = extra_pars["gamma"]
        self.t0 = extra_pars["t0"]
        self.kappa = extra_pars["kappa"]
        self.delta = extra_pars["delta"]
        self.adapt_epsilon = extra_pars["adapt_epsilon"]
        if 0 <= extra_pars["adapt_epsilon"] <= 1:
            self.adapt_epsilon = np.int(np.round(self.adapt_epsilon * self.iter))

    def leapfrog(self, position, momentum):
        momentum -= self.epsilon * self.grad_U(position) / 2
        position += self.epsilon * momentum
        momentum -= self.epsilon * self.grad_U(position) / 2

        return [position, momentum]

    def longest_batch(self, position):
        current_position = deepcopy(position)
        momentum = np.random.normal(size=self.dim)
        current_momentum = deepcopy(momentum)
        increment = stop = stop_ind = iter_batch = 0

        while increment >= 0 or iter_batch < self.L0:
            iter_batch += 1
            [position, momentum] = self.leapfrog(position, momentum)

            if stop == 0:
                position_diff = position - current_position
                increment = np.dot(position_diff, momentum)
                if np.isnan(increment):
                    return [current_position, iter_batch, True]
                if increment < 0:
                    stop_ind = iter_batch
                    stop = 1
            if iter_batch == self.L0 - 1:
                break
            if iter_batch > 1e4:
                return [current_position, iter_batch, True]

        momentum = - momentum
        current_U = self.U(current_position)
        current_K = np.sum(current_momentum ** 2) / 2
        proposed_U = self.U(position)
        proposed_K = np.sum(momentum ** 2) / 2

        alpha = current_U + current_K - proposed_U - proposed_K
        if np.isnan(alpha):
            return [current_position, None, True]
        else:
            if np.log(np.random.uniform(size=1)) < alpha:
                return [position, stop_ind, False]
            else:
                return [current_position, stop_ind, False]

    def learnL(self):
        self.L.append(self.L0)
        position = deepcopy(self.current_position)
        for i in range(self.adapt_L):
            L = np.quantile(self.L, 0.95, interpolation="nearest")
            L = 2 if L <= 1 else L
            self.L0 = np.random.randint(low=1, high=L)
            [position, L_emp, diverge] = self.longest_batch(position)
            if not diverge:
                self.L.append(L_emp)

    def adaptation(self):
        frac = 1 / (self.curr_iter + self.t0)
        self.h_bar = (1 - frac) * self.h_bar + frac * (self.delta - self.alpha)
        self.epsilon_iter = np.array(np.exp(self.mu - (np.sqrt(self.curr_iter) / self.gamma) * self.h_bar)).reshape(
            -1, )
        eta = self.curr_iter ** -self.kappa
        self.epsilon_bar = np.exp(eta * np.log(self.epsilon_iter) + (1 - eta) * np.log(self.epsilon_bar))

    def step(self):

        u = np.random.uniform(size=1)

        if self.dual_averaging and self.curr_iter <= self.adapt_epsilon:
            self.epsilon = self.epsilon_iter
        else:
            self.epsilon = np.random.uniform(low=self.epsilon_iter * (1 - self.epsilon_noise),
                                             high=self.epsilon_iter * (1 + self.epsilon_noise), size=1)

        if u < self.refreshment_prob:
            self.l = self.L0 + 1

            self.w_plus = list()
            self.current_momentum = np.random.normal(size=self.dim)
            self.w_plus.append([self.current_position, self.current_momentum])

            momentum = deepcopy(self.current_momentum)
            position = deepcopy(self.current_position)

            for i in range(self.L0):
                [position, momentum] = self.leapfrog(position, momentum)
                self.w_plus += list([[position, momentum]])
                # positions[i, :] = position
                # momentums[i, :] = momentum

            self.grad_evals += self.L0
            self.traj_length.append(self.L0)

            pos_new = self.w_plus[self.l - 1][0]
            mom_new = self.w_plus[self.l - 1][1]
            acc_ratio = compute_acc_ratio(self.current_position, self.current_momentum, pos_new, mom_new, self.U)


            self.alpha = np.min((1, acc_ratio))
            if np.random.uniform(size=1) < self.alpha:
                self.current_position = pos_new
                self.current_momentum = mom_new
                self.current_i = self.l
                self.direction = 1
            else:
                self.current_momentum = -self.current_momentum
                self.current_i = 1
                self.direction = -1
        else:
            j = self.current_i + self.direction * self.L0
            delta = j - self.l if self.direction == 1 else -(j - 1)

            if delta > 0:
                self.l += delta

                momentum = deepcopy(self.current_momentum)
                position = deepcopy(self.current_position)
                self.epsilon = - self.epsilon
                for i in range(delta):
                    [position, momentum] = self.leapfrog(position, momentum)
                    if self.direction == 1:
                        self.w_plus += list([[position, momentum]])
                    else:
                        self.w_plus = list([[position, -momentum]]) + self.w_plus

                if self.direction == -1:
                    self.current_i += delta
                    j = 1


            pos_new = self.w_plus[j - 1][0]
            mom_new = self.w_plus[j - 1][1]
            acc_ratio = compute_acc_ratio(self.current_position, self.current_momentum, pos_new, -self.direction * mom_new, self.U)

            self.alpha = np.min((1, acc_ratio))
            if np.random.uniform(size=1) < self.alpha:
                self.current_position = pos_new
                self.current_momentum = self.direction * mom_new
                self.current_i = j
            else:
                self.current_momentum = -self.current_momentum
                self.direction = -self.direction

        if self.dual_averaging:
            if self.curr_iter <= self.adapt_epsilon:
                self.adaptation()
            else:
                self.epsilon_iter = self.epsilon_bar

    def simulate(self):
        results = np.zeros((self.iter + self.adapt_epsilon, self.dim))
        t0 = process_time()
        self.direction = 1
        self.current_i = 1
        self.l = 1
        self.current_momentum = np.random.normal(size=self.dim)
        self.w_plus.append([self.current_position, self.current_momentum])

        if self.adapt_with_opt_epsilon:
            start_pos = deepcopy(self.current_position)
            start_mom = deepcopy(self.current_momentum)

        if self.dual_averaging:
            curr_pos = deepcopy(self.current_position)
            self.epsilon_iter = find_reasonable_epsilon(curr_pos, self.grad_U, self.U)
            self.mu = np.log(10. * self.epsilon_iter)
            self.epsilon = self.epsilon_iter

        if not self.adapt_with_opt_epsilon:
            self.learnL()

        for i in range(self.iter + self.adapt_epsilon):
            if i > self.adapt_epsilon:
                if self.version == "vanilla":
                    self.L0 = np.random.choice(self.L)
                if self.version == "quantile":
                    self.L0 = np.quantile(self.L, self.quantile, interpolation="nearest")
                if self.version == "uniform":
                    self.L0 = np.random.choice(self.L)
                    self.L0 = 2 if self.L0 <= 1 else self.L0
                    self.L0 = np.random.randint(low=1, high=self.L0)
                self.L0 = np.max((1, np.int(self.L0/3)))
            self.step()
            results[i, :] = self.current_position

            if self.adapt_with_opt_epsilon:
                if i == self.adapt_epsilon - 1:
                    self.learnL()
                    self.current_position = start_pos
                    self.current_momentum = start_mom

            self.curr_iter += 1
        t1 = process_time()
        self.elapsed_time = t1 - t0
        return np.array(results)

"""
Original code for No-U-Turn Sampler (NUTS) and
Hamiltonian Monte Carlo (HMC) by Mat Leonard:
https://github.com/mcleonard/sampyl
"""
class NUTS():

    def __init__(self, U, grad_U, start_position, extra_pars, iter):
        self.U = U
        self.grad_U = grad_U
        self.start_position = start_position
        self.iter = iter
        self.dim = len(start_position)
        self.epsilon = None
        self.L = list()
        self.grad_evals = 0
        self.h_bar = 0
        self.mu = self.epsilon_bar = self.gamma = self.t0 = self.kappa = self.delta = self.adapt_epsilon = None
        self.set_dual_averaging_pars(extra_pars)
        self.elapsed_time = 0
        self.epsilon_iter = extra_pars["start_epsilon"]
        self.delta_max = extra_pars["delta_max"]
        self.elapsed_time = 0
        self.L = list()
        self.grad_evals = 0

    def set_dual_averaging_pars(self, extra_pars):
        self.epsilon_bar = extra_pars["epsilon_bar"]
        self.gamma = extra_pars["gamma"]
        self.t0 = extra_pars["t0"]
        self.kappa = extra_pars["kappa"]
        self.delta = extra_pars["delta"]
        self.adapt_epsilon = extra_pars["adapt_epsilon"]
        if 0 <= extra_pars["adapt_epsilon"] <= 1:
            self.adapt_epsilon = np.int(np.round(self.adapt_epsilon * self.iter))

    def bern(self, p):
        return np.random.uniform() < p

    def leapfrog(self, x, r, step_size):
        r1 = r - step_size / 2 * self.grad_U(x)
        x1 = x + step_size * r1
        r2 = r1 - step_size / 2 * self.grad_U(x1)
        return x1, r2

    def energy(self, x, r):
        return -self.U(x) - 0.5 * np.dot(r, r)

    def buildtree(self, x, r, u, v, j, e, x0, r0):
        if j == 0:
            x1, r1 = self.leapfrog(x, r, v * e)
            self.grad_evals += 1
            E = self.energy(x1, r1)
            E0 = self.energy(x0, r0)
            dE = E - E0

            n1 = (np.log(u) - dE <= 0)
            s1 = (np.log(u) - dE < self.delta_max)
            if dE < -30:
                dE = -30
            elif dE > 30:
                dE = 30
            return x1, r1, x1, r1, x1, n1, s1, np.min(np.array([1, np.exp(dE)])), 1
        else:
            xn, rn, xp, rp, x1, n1, s1, a1, na1 = \
                self.buildtree(x, r, u, v, j - 1, e, x0, r0)
            if s1 == 1:
                if v == -1:
                    xn, rn, _, _, x2, n2, s2, a2, na2 = \
                        self.buildtree( xn, rn, u, v, j - 1, e, x0, r0)
                else:
                    _, _, xp, rp, x2, n2, s2, a2, na2 = \
                        self.buildtree( xp, rp, u, v, j - 1, e, x0, r0)
                if self.bern(n2 / max(n1 + n2, 1.)):
                    x1 = x2

                a1 = a1 + a2
                na1 = na1 + na2

                dx = xp - xn
                s1 = s2 * (np.dot(dx, rn) >= 0) * \
                     (np.dot(dx, rp) >= 0)
                n1 = n1 + n2
            return xn, rn, xp, rp, x1, n1, s1, a1, na1

    def simulate(self):
        sample_steps = self.iter
        adapt_steps = self.adapt_epsilon

        curr_pos = deepcopy(self.start_position)
        self.epsilon_iter = find_reasonable_epsilon(curr_pos, self.grad_U, self.U)
        mu = np.log(10 * self.epsilon_iter)
        dims = len(self.start_position)
        scale = np.ones(dims)

        Hbar = self.h_bar
        ebar = self.epsilon_bar

        samples = np.zeros((sample_steps, dims))
        x = self.start_position.copy()

        start_time = process_time()
        for i in range(sample_steps + adapt_steps):
            # do NUTS step
            r0 = np.random.multivariate_normal(np.zeros(dims), np.diagflat(scale))
            u = np.random.uniform()
            e = self.epsilon_iter
            xn, xp, rn, rp, y = x, x, r0, r0, x
            j, n, s = 0, 1, 1
            natot = 0
            while s == 1:
                v = self.bern(0.5) * 2 - 1
                if v == -1:
                    xn, rn, _, _, x1, n1, s1, a, na = self.buildtree(xn, rn, u, v, j, e, x, r0)
                else:
                    _, _, xp, rp, x1, n1, s1, a, na = self.buildtree(xp, rp, u, v, j, e, x, r0)

                if s1 == 1 and self.bern(np.min(np.array([1, n1 / n]))):
                    y = x1

                dx = xp - xn
                s = s1 * (np.dot(dx, rn) >= 0) * (np.dot(dx, rp) >= 0)
                n = n + n1
                j = j + 1
                natot += na

            if i >= adapt_steps:
                self.epsilon_iter = ebar
            else:
                # Adapt step size
                w = 1. / (i + 1. + self.t0)
                Hbar = (1 - w) * Hbar + w * (self.delta - a / na)
                log_e = mu - ((i + 1.) ** .5 / self.gamma) * Hbar
                self.epsilon_iter = np.exp(log_e)
                z = (i + 1.) ** (-self.kappa)
                ebar = np.exp(z * log_e + (1 - z) * np.log(ebar))

            x[:] = y

            if i >= adapt_steps:
                samples[i - adapt_steps, :] = x
                self.L.append(natot)
        end_time = process_time()
        self.elapsed_time = end_time - start_time

        return samples