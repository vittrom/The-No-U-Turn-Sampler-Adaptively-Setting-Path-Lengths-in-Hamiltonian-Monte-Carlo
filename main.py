from models import *
from hmc import *
from metrics import *

volatility = stochVolatility()
U, grad_U = volatility.params()

dim = 1003

start_position = np.repeat(1., dim) #np.repeat(3., 100)  #.array([-1.5, -1.55])
iter = 100
samp_random = True
dual_averaging = True

### HMC ###
print("Simulating wiggle HMC")
extra_pars_wiggle = dict({
    "L_start": 25,
    "L_noise": 0.2,
    "threshold": 180,
    "epsilon_noise": 0.2,
    "epsilon_start": 0.25,
    "version": "distr_L",
    "adapt_L": 100,
    "fn": np.mean,
    "method": "quantile",
    "quantile_ub": 0.95,
    "quantile_lb": 0.05,
    "epsilon_bar": 1,
    "gamma": 0.05,
    "t0": 10,
    "kappa": 0.75,
    "delta": 0.65,
    "adapt_epsilon": 100
})
hmc_wiggle_test = HMC_wiggle(U, grad_U, start_position, extra_pars_wiggle, iter, samp_random, dual_averaging)
res_wiggleHMC = hmc_wiggle_test.simulate()
start = hmc_wiggle_test.adapt_epsilon
res_hmc = res_wiggleHMC[start:, :]

print(res_hmc[-1:, :])
print("HMC:" + str(ESJD(res_hmc)/hmc_wiggle_test.grad_evals))
