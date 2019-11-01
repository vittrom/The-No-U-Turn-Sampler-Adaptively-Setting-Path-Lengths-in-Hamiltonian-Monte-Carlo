from hmc import *
import matplotlib.pyplot as plt
from metrics import *
from mvn import *

mvn = mvn(100)
U, grad_U = mvn.params()

### Test cases for all the methods
start_position = np.repeat(3., 100)  #.array([-1.5, -1.55])
iter = 1000
samp_random = True
dual_averaging = True

### HMC ###
print("Simulating HMC")
extra_pars_HMC = dict({
    "L_start": 10,
    "L_noise": 0.2,
    "epsilon_noise": 0.2,
    "epsilon_start": 0.25,
    "epsilon_bar": 1,
    "gamma": 0.05,
    "t0": 10,
    "kappa": 0.75,
    "delta": 0.65,
    "adapt_epsilon": 100
})
hmc_test = HMC(U, grad_U, start_position, extra_pars_HMC, iter, samp_random, dual_averaging)
res_hmc = hmc_test.simulate()
start = hmc_test.adapt_epsilon
res_hmc = res_hmc[start:, :]

### HMC wiggle ###
'''
vanilla: no extra parameters to specify
vanish_vanilla: specify function
distr_L: specify method
            - quantile: specify LB, UB
            - random: nothing to specify
            - random_unif: nothing to specify
'''
print("Simulating wiggle HMC")
extra_pars_wiggle = dict({
    "L_start": 10,
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
res_wiggleHMC = res_wiggleHMC[start:, :]

### eHMC ###
print("Simulating eHMC")
extra_pars_eHMC = dict({
    "L_start": 10,
    "L_noise": 0.2,
    "threshold": 180,
    "epsilon_noise": 0.2,
    "epsilon_start": 0.25,
    "version": "vanilla",
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
eHMC_test = eHMC(U, grad_U, start_position, extra_pars_eHMC, iter, samp_random, dual_averaging)
res_eHMC = eHMC_test.simulate()
res_eHMC = res_eHMC[start:, :]

### prHMC ###
print("Simulating prHMC")
extra_pars_prHMC = dict({
    "L_start": 10,
    "L_noise": 0.2,
    "threshold": 180,
    "epsilon_noise": 0.2,
    "epsilon_start": 0.25,
    "version": "vanilla",
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
prHMC_test = prHMC(U, grad_U, start_position, extra_pars_prHMC, iter, dual_averaging)
res_prHMC = prHMC_test.simulate()
res_prHMC = res_prHMC[start:, :]

print("Expexted squared jump distance per gradient")
print("HMC:" + str(ESJD(res_hmc)/hmc_test.grad_evals))
print("Wiggle:" + str(ESJD(res_wiggleHMC)/hmc_wiggle_test.grad_evals))
print("eHMC:" + str(ESJD(res_eHMC)/eHMC_test.grad_evals))
print("prHMC:" + str(ESJD(res_prHMC)/prHMC_test.grad_evals))

print("Effective sample size per second")
print("HMC:" + str([ESS(res_hmc[:, i], 100)/hmc_test.elapsed_time for i in range(res_hmc.shape[1])]))
print("Wiggle:" + str([ESS(res_wiggleHMC[:, i], 100)/hmc_wiggle_test.elapsed_time for i in range(res_wiggleHMC.shape[1])]))
print("eHMC:" + str([ESS(res_eHMC[:, i], 100)/eHMC_test.elapsed_time for i in range(res_eHMC.shape[1])]))
print("prHMC:" + str([ESS(res_prHMC[:, i], 100)/prHMC_test.elapsed_time for i in range(res_prHMC.shape[1])]))

print("Effective sample size per gradient")
print("HMC:" + str([ESS(res_hmc[:, i], 100)/hmc_test.grad_evals for i in range(res_hmc.shape[1])]))
print("Wiggle:" + str([ESS(res_wiggleHMC[:, i], 100)/hmc_wiggle_test.grad_evals for i in range(res_wiggleHMC.shape[1])]))
print("eHMC:" + str([ESS(res_eHMC[:, i], 100)/eHMC_test.grad_evals for i in range(res_eHMC.shape[1])]))
print("prHMC:" + str([ESS(res_prHMC[:, i], 100)/prHMC_test.grad_evals for i in range(res_prHMC.shape[1])]))