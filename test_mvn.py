from methods import *
import matplotlib.pyplot as plt
from models import *
import os

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
            "adapt_epsilon": 100
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
            "adapt_epsilon": 100
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

dim = 2
off_diag = 0.95
mvn = mvn(dim=dim, off_diag=off_diag)
U, grad_U = mvn.params()

start_position = np.array([-1.5, -1.55]) #np.repeat(1., dim)
iter = 1000
samp_random = True
dual_averaging = True

nuts_test = HMC_wiggle(U, grad_U, start_position, extra_pars_wiggle, iter)#, samp_random, dual_averaging)
res_wiggleHMC = nuts_test.simulate()
start = nuts_test.adapt_epsilon
res_hmc = res_wiggleHMC

print(res_hmc.shape)
print(np.mean(nuts_test.L))
# plt.hist(nuts_test.L)
# plt.xlabel("Leapfrog steps")
# plt.ylabel("Count")
# plt.title("Histogram of leapfrog steps from NWHMC")
# plt.savefig(os.path.join("Plots","mvn_example.pdf"))

plt.figure(1)
plt.scatter(res_hmc[:, 0], res_hmc[:, 1], s=15)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("NWHMC")
plt.savefig(os.path.join("Plots", "nwhmc_samps.pdf"))