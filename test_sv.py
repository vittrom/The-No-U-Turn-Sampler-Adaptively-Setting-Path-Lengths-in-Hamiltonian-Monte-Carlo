from methods import *
import matplotlib.pyplot as plt
from metrics import *
from models import *

extra_pars_eHMC = dict({
            "L_start": 10,
            "L_noise": 0.2,
            "epsilon_noise": 0.2,
            "epsilon_start": 0.25,
            "version": "uniform",
            "adapt_L": 100,
            "quantile_ub": 0.95,
            "adapt_with_opt_epsilon": True,
            "epsilon_bar": 1,
            "gamma": 0.05,
            "t0": 10,
            "kappa": 0.75,
            "delta": 0.65,
            "adapt_epsilon": 100
        })
extra_pars_prHMC = dict({
            "L_start": 10,
            "L_noise": 0.2,
            "epsilon_noise": 0.2,
            "epsilon_start": 0.25,
            "version": "uniform",
            "adapt_L": 100,
            "quantile_ub": 0.95,
            "adapt_with_opt_epsilon": True,
            "refreshment_prob": 0.4,
            "epsilon_bar": 1,
            "gamma": 0.05,
            "t0": 10,
            "kappa": 0.75,
            "delta": 0.65,
            "adapt_epsilon": 100
        })
extra_pars_HMC = dict({
            "L_start": 25,
            "L_noise": 0.2,
            "epsilon_noise": 0.2,
            "epsilon_start": 0.25,
            "epsilon_bar": 1,
            "gamma": 0.05,
            "t0": 10,
            "kappa": 0.75,
            "delta": 0.65,
            "adapt_epsilon": 10
        })
extra_pars_wiggle = dict({
            "L_start": 25,
            "L_noise": 0.2,
            "threshold": 180,
            "epsilon_noise": 0.2,
            "epsilon_start": 0.25,
            "version": "vanilla",
            "adapt_L": 100,
            "fn": np.mean,
            "method": "quantile",
            "quantile_ub": 0.95,
            "quantile_lb": 0.00,
            "epsilon_bar": 1,
            "gamma": 0.05,
            "t0": 10,
            "kappa": 0.75,
            "delta": 0.65,
            "adapt_epsilon": 10
        })
extra_pars_NUTS = dict({
            "epsilon_bar": 1,
            "gamma": 0.05,
            "t0": 10,
            "kappa": 0.75,
            "delta": 0.65,
            "adapt_epsilon": 10,
            "delta_max": 1000,
            "start_epsilon": 0.25
        })

volatility = stochVolatility()
U, grad_U = volatility.params()

dims = 1003
start_position = np.random.normal(size=dims)
iter = 10
samp_random = True
dual_averaging = True

hmc_wiggle_test = HMC_wiggle(U, grad_U, start_position, extra_pars_wiggle, iter, samp_random,dual_averaging)
res_wiggleHMC = hmc_wiggle_test.simulate()
start = hmc_wiggle_test.adapt_epsilon
print(res_wiggleHMC.shape)
res_hmc = res_wiggleHMC[start:, :]

print(res_hmc)
print("HMC:" + str(ESJD(res_hmc)/hmc_wiggle_test.grad_evals))
plt.figure(1)
plt.scatter(res_hmc[:, 0], res_hmc[:, 1])
plt.show()

nuts_test = NUTS(U, grad_U, start_position, extra_pars_NUTS, iter)
res_wiggleHMC = nuts_test.simulate()
start = nuts_test.adapt_epsilon
res_hmc = res_wiggleHMC
print(res_hmc.shape)
plt.figure(2)
plt.scatter(res_hmc[:, 0], res_hmc[:,1])
plt.show()