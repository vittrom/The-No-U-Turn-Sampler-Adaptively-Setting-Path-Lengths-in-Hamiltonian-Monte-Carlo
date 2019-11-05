from hmc import *
import matplotlib.pyplot as plt
from metrics import *
from models import *
from nuts import *
import os

dim =100
mvn = mvn(dim=dim, off_diag=0.99)
U, grad_U = mvn.params()

### Test cases for all the methods
start_position = np.repeat(3., dim) #np.repeat(3., 100)  #.array([-1.5, -1.55])
iter = 1000
samp_random = True
dual_averaging = True
delta = np.array((0.75, 0.8, 0.85, 0.9, 0.95)) #0.6, 0.65, 0.7,
replications = 10
lag = 100


for d in delta:
    print(d)
    esjd_l_hmc = np.zeros((replications, 2))
    ess_l_hmc = np.zeros((replications, dim + 1))
    ess_s_hmc = np.zeros((replications, dim + 1))

    esjd_l_wiggle_hmc = np.zeros((replications, 2))
    ess_l_wiggle_hmc = np.zeros((replications, dim + 1))
    ess_s_wiggle_hmc = np.zeros((replications, dim + 1))

    esjd_l_ehmc = np.zeros((replications, 2))
    ess_l_ehmc = np.zeros((replications, dim + 1))
    ess_s_ehmc = np.zeros((replications, dim + 1))

    esjd_l_prhmc = np.zeros((replications, 2))
    ess_l_prhmc = np.zeros((replications, dim + 1))
    ess_s_prhmc = np.zeros((replications, dim + 1))

    esjd_l_nuts = np.zeros((replications, 2))
    ess_l_nuts = np.zeros((replications, dim + 1))
    ess_s_nuts = np.zeros((replications, dim + 1))

    for i in range(replications):
        print(i)
        ### HMC ###
        print("Simulating HMC")
        extra_pars_HMC = dict({
            "L_start": 25,
            "L_noise": 0.2,
            "epsilon_noise": 0.2,
            "epsilon_start": 0.25,
            "epsilon_bar": 1,
            "gamma": 0.05,
            "t0": 10,
            "kappa": 0.75,
            "delta": d,
            "adapt_epsilon": 100
        })
        hmc_test = HMC(U, grad_U, start_position, extra_pars_HMC, iter, samp_random, dual_averaging)
        res_hmc = hmc_test.simulate()
        start = hmc_test.adapt_epsilon
        res_hmc = res_hmc[start:, :]

        #Compute metrics
        esjd_l_hmc[i, :] = np.array((ESJD(res_hmc)/hmc_test.grad_evals, d))
        ess_s_hmc[i, :] = [ESS(res_hmc[:, i], lag)/hmc_test.elapsed_time for i in range(res_hmc.shape[1])] + [d]
        ess_l_hmc[i, :] = [ESS(res_hmc[:, i], lag)/hmc_test.grad_evals for i in range(res_hmc.shape[1])] + [d]

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
            "delta": d,
            "adapt_epsilon": 100
        })
        hmc_wiggle_test = HMC_wiggle(U, grad_U, start_position, extra_pars_wiggle, iter, samp_random, dual_averaging)
        res_wiggleHMC = hmc_wiggle_test.simulate()
        res_wiggleHMC = res_wiggleHMC[start:, :]
        #Compute metrics
        esjd_l_wiggle_hmc[i, :] = np.array((ESJD(res_wiggleHMC)/hmc_wiggle_test.grad_evals, d))
        ess_s_wiggle_hmc[i, :] = [ESS(res_wiggleHMC[:, i], 100)/hmc_wiggle_test.elapsed_time for i in range(res_wiggleHMC.shape[1])] + [d]
        ess_l_wiggle_hmc[i, :] = [ESS(res_wiggleHMC[:, i], 100)/hmc_wiggle_test.grad_evals for i in range(res_wiggleHMC.shape[1])] + [d]

        ### eHMC ###
        print("Simulating eHMC")
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
            "delta": d,
            "adapt_epsilon": 100
        })
        eHMC_test = eHMC(U, grad_U, start_position, extra_pars_eHMC, iter, samp_random, dual_averaging)
        res_eHMC = eHMC_test.simulate()
        res_eHMC = res_eHMC[start:, :]

        #Compute metrics
        esjd_l_ehmc[i, :] = np.array((ESJD(res_eHMC) / eHMC_test.grad_evals, d))
        ess_s_ehmc[i, :] = [ESS(res_eHMC[:, i], 100) / eHMC_test.elapsed_time for i in range(res_eHMC.shape[1])] + [d]
        ess_l_ehmc[i, :] = [ESS(res_eHMC[:, i], 100) / eHMC_test.grad_evals for i in range(res_eHMC.shape[1])] + [d]

        ### prHMC ###
        print("Simulating prHMC")
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
            "delta": d,
            "adapt_epsilon": 100
        })
        prHMC_test = prHMC(U, grad_U, start_position, extra_pars_prHMC, iter, dual_averaging)
        res_prHMC = prHMC_test.simulate()
        res_prHMC = res_prHMC[start:, :]

        # Compute metrics
        esjd_l_prhmc[i, :] = np.array((ESJD(res_prHMC) / prHMC_test.grad_evals, d))
        ess_s_prhmc[i, :] = [ESS(res_prHMC[:, i], 100) / prHMC_test.elapsed_time for i in range(res_prHMC.shape[1])] + [d]
        ess_l_prhmc[i, :] = [ESS(res_prHMC[:, i], 100) / prHMC_test.grad_evals for i in range(res_prHMC.shape[1])] + [d]

        # #ADD NUTS
        print("Simulating NUTS")
        extra_pars_nuts = dict({
            "epsilon_bar": 1,
            "gamma": 0.05,
            "t0": 10,
            "kappa": 0.75,
            "delta": d,
            "adapt_epsilon": 100
        })
        nuts_test = NUTS(U, grad_U, start_position, extra_pars_nuts, iter)
        res_nuts = nuts_test.simulate()

        # Compute metrics
        esjd_l_nuts[i, :] = np.array((ESJD(res_nuts) / nuts_test.grad_evals, d))
        ess_s_nuts[i, :] = [ESS(res_nuts[:, i], 100) / nuts_test.elapsed_time for i in range(res_nuts.shape[1])] + [d]
        ess_l_nuts[i, :] = [ESS(res_nuts[:, i], 100) / nuts_test.grad_evals for i in range(res_nuts.shape[1])] + [d]

    cur_d = str(d).replace(".", "_")
    np.savez(os.path.join("Results", "HMC", "results_" + cur_d + ".npz"), esjd_l_hmc, ess_s_hmc, ess_l_hmc)
    np.savez(os.path.join("Results", "wiggle_HMC", "results_" + cur_d + ".npz"), esjd_l_wiggle_hmc, ess_s_wiggle_hmc, ess_l_wiggle_hmc)
    np.savez(os.path.join("Results", "eHMC", "results_" + cur_d + ".npz"), esjd_l_ehmc, ess_s_ehmc, ess_l_ehmc)
    np.savez(os.path.join("Results", "prHMC", "results_" + cur_d + ".npz"), esjd_l_prhmc, ess_s_prhmc, ess_l_prhmc)
    np.savez(os.path.join("Results", "NUTS", "results_" + cur_d + ".npz"), esjd_l_nuts, ess_s_nuts, ess_l_nuts)
