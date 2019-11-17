from methods import *
import matplotlib.pyplot as plt
from metrics import *
from models import *
from statsmodels.tsa.stattools import acf

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
            "delta": 0.8,
            "adapt_epsilon": 10
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
            "version": "distr_L",
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
            "delta": 0.8,
            "adapt_epsilon": 10,
            "delta_max": 1000,
            "start_epsilon": 0.25
        })

log_reg = logReg()
U, grad_U = log_reg.params()

dims = 25
start_position = np.repeat(0., dims) #np.random.normal(size=dims)
iter = 500
samp_random = True
dual_averaging = True

hmc_wiggle_test = eHMC(U, grad_U, start_position, extra_pars_eHMC, iter, samp_random, dual_averaging)
res_wiggleHMC = hmc_wiggle_test.simulate()
start = hmc_wiggle_test.adapt_epsilon
res_hmc = res_wiggleHMC[start:, :]
print(hmc_wiggle_test.elapsed_time)
print(acf(res_hmc[:, 1]))
print(acf(res_hmc[:, 1]).sum())
print("HMC:" + str(ESJD(res_hmc)/hmc_wiggle_test.grad_evals))
print("HMC:" + str(ESS(res_hmc[:,1])/hmc_wiggle_test.grad_evals))


# nuts_test = NUTS(U, grad_U, start_position, extra_pars_NUTS, iter)
# res_wiggleHMC = nuts_test.simulate()
# start = nuts_test.adapt_epsilon
# res_hmc = res_wiggleHMC
# print(res_hmc.shape)
