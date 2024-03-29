from methods import *
import matplotlib.pyplot as plt
from metrics import *
from models import *
from plots import *
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
            "adapt_epsilon": 100
        })
extra_pars_NUTS = dict({
            "epsilon_bar": 1,
            "gamma": 0.05,
            "t0": 10,
            "kappa": 0.75,
            "delta": 0.65,
            "adapt_epsilon": 100,
            "delta_max": 1000,
            "start_epsilon": 0.25
        })

dims = 2
B = 0.1
banana = banana(dims=dims, B=B)
U, grad_U = banana.params()

start_position = np.repeat(1., dims) #np.repeat(3., 100)  #.array([-1.5, -1.55])
iter = 1000
samp_random = True
dual_averaging = False

hmc_wiggle_test = HMC_wiggle(U, grad_U, start_position, extra_pars_wiggle, iter, samp_random, dual_averaging)
res_wiggleHMC = hmc_wiggle_test.simulate()
start = hmc_wiggle_test.adapt_epsilon
print(res_wiggleHMC.shape)
res_hmc = res_wiggleHMC[start:, :]

print(hmc_wiggle_test.elapsed_time/hmc_wiggle_test.grad_evals)
print(ESS(res_hmc[:, 1]))
print(1000*ESJD(res_hmc))
print(hmc_wiggle_test.grad_evals)

plot_autocorr(res_hmc, "NWHMC", 1, os.path.join("Plots", "wiggle_HMC", "banana", "autocorr", "corr.png"), lag=100)

# print(res_hmc)
# print("HMC:" + str(ESJD(res_hmc)/hmc_wiggle_test.grad_evals))
# plt.figure(1)
# plt.scatter(res_hmc[:, 0], res_hmc[:, 1])
# plt.show()

nuts_test = NUTS(U, grad_U, start_position, extra_pars_NUTS, iter)
res_wiggleHMC = nuts_test.simulate()
start = nuts_test.adapt_epsilon
res_hmc = res_wiggleHMC
print(res_hmc.shape)
print(nuts_test.elapsed_time/nuts_test.grad_evals)
print(ESS(res_hmc[:,1]))
print(1000*ESJD(res_hmc))
print(nuts_test.grad_evals)
plot_autocorr(res_hmc, "NUTS", 1, os.path.join("Plots", "NUTS", "banana", "autocorr", "corr.png"), lag=100)

# plt.figure(2)
# plt.scatter(res_hmc[:, 0], res_hmc[:,1])
# plt.show()