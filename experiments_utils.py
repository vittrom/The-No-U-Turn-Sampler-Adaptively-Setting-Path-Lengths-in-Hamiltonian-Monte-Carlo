from methods import *
from plots import *
from metrics import *
from utils import *
import os

def run_experiment(U, grad_U, start_position, iter, name, delta, replications, dim, extra_pars, lag=100):
    L_start, L_noise, epsilon_noise, epsilon_start, epsilon_bar, gamma, t0, kappa, adapt_epsilon, \
    threshold, quantile_ub, quantile_lb, adapt_L, adapt_opt_epsilon, \
    refreshment_prob, delta_max, dual_averaging, samp_random = set_extra_pars(extra_pars)

    start = adapt_epsilon

    for d in delta:
        print(d)
        esjd_l_hmc, ess_l_hmc, ess_s_hmc, esjd_l_wiggle_hmc, ess_l_wiggle_hmc, \
        ess_s_wiggle_hmc, esjd_l_ehmc, ess_l_ehmc, ess_s_ehmc, esjd_l_prhmc, \
        ess_l_prhmc, ess_s_prhmc, esjd_l_nuts, ess_l_nuts, ess_s_nuts = reset_experiment_matrix(replications, dim)

        curr_d = str(d).replace(".", "_")
        for i in range(replications):
            print(i)
            ### HMC ###
            print("Simulating HMC")
            extra_pars_HMC = dict({
                "L_start": L_start,
                "L_noise": L_noise,
                "epsilon_noise": epsilon_noise,
                "epsilon_start": epsilon_start,
                "epsilon_bar": epsilon_bar,
                "gamma": gamma,
                "t0": t0,
                "kappa": kappa,
                "delta": d,
                "adapt_epsilon": adapt_epsilon
            })
            hmc_test = HMC(U, grad_U, start_position, extra_pars_HMC, iter, samp_random, dual_averaging)
            res_hmc = hmc_test.simulate()
            res_hmc = res_hmc[start:, :]

            # Compute metrics
            esjd_l_hmc[i, :] = np.array((ESJD(res_hmc) / hmc_test.grad_evals, d))
            ess_s_hmc[i, :] = [ESS(res_hmc[:, i], lag) / hmc_test.elapsed_time for i in range(res_hmc.shape[1])] + [d]
            ess_l_hmc[i, :] = [ESS(res_hmc[:, i], lag) / hmc_test.grad_evals for i in range(res_hmc.shape[1])] + [d]

            # Plot autocorrelation for one of the dimensions
            col = np.min((5, dim - 1))
            plot_autocorr(res_hmc, col,
                          os.path.join("Plots", "HMC", name, "autocorr", "autocorr_delta" + curr_d + "_dim_" + str(dim) + "_rep_" + str(i) + ".png"),
                          lag=lag)

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
                "L_start": L_start,
                "L_noise": L_noise,
                "threshold": threshold,
                "epsilon_noise": epsilon_noise,
                "epsilon_start": epsilon_start,
                "version": "distr_L",
                "adapt_L": adapt_L,
                "fn": np.mean,
                "method": "quantile",
                "quantile_ub": quantile_ub,
                "quantile_lb": quantile_lb,
                "epsilon_bar": epsilon_bar,
                "gamma": gamma,
                "t0": t0,
                "kappa": kappa,
                "delta": d,
                "adapt_epsilon": adapt_epsilon
            })
            hmc_wiggle_test = HMC_wiggle(U, grad_U, start_position, extra_pars_wiggle, iter, samp_random,
                                         dual_averaging)
            res_wiggleHMC = hmc_wiggle_test.simulate()
            res_wiggleHMC = res_wiggleHMC[start:, :]
            # Compute metrics
            esjd_l_wiggle_hmc[i, :] = np.array((ESJD(res_wiggleHMC) / hmc_wiggle_test.grad_evals, d))
            ess_s_wiggle_hmc[i, :] = [ESS(res_wiggleHMC[:, i], 100) / hmc_wiggle_test.elapsed_time for i in
                                      range(res_wiggleHMC.shape[1])] + [d]
            ess_l_wiggle_hmc[i, :] = [ESS(res_wiggleHMC[:, i], 100) / hmc_wiggle_test.grad_evals for i in
                                      range(res_wiggleHMC.shape[1])] + [d]

            # Plot autocorrelation for one of the dimensions
            plot_autocorr(res_wiggleHMC, col,
                          os.path.join("Plots","wiggle_HMC",name,  "autocorr", "autocorr_delta" + curr_d + "_dim_" + str(dim) + "_rep_" + str(i) + ".png"),
                          lag=lag)

            ### eHMC ###
            print("Simulating eHMC")
            extra_pars_eHMC = dict({
                "L_start": L_start,
                "L_noise": L_noise,
                "epsilon_noise": epsilon_noise,
                "epsilon_start": epsilon_start,
                "version": "uniform",
                "adapt_L": adapt_L,
                "quantile_ub": quantile_ub,
                "adapt_with_opt_epsilon": adapt_opt_epsilon,
                "epsilon_bar": epsilon_bar,
                "gamma": gamma,
                "t0": t0,
                "kappa": kappa,
                "delta": d,
                "adapt_epsilon": adapt_epsilon
            })
            eHMC_test = eHMC(U, grad_U, start_position, extra_pars_eHMC, iter, samp_random, dual_averaging)
            res_eHMC = eHMC_test.simulate()
            res_eHMC = res_eHMC[start:, :]

            # Compute metrics
            esjd_l_ehmc[i, :] = np.array((ESJD(res_eHMC) / eHMC_test.grad_evals, d))
            ess_s_ehmc[i, :] = [ESS(res_eHMC[:, i], 100) / eHMC_test.elapsed_time for i in range(res_eHMC.shape[1])] + [
                d]
            ess_l_ehmc[i, :] = [ESS(res_eHMC[:, i], 100) / eHMC_test.grad_evals for i in range(res_eHMC.shape[1])] + [d]

            # Plot autocorrelation for one of the dimensions
            plot_autocorr(res_eHMC, col,
                          os.path.join("Plots", "eHMC", name, "autocorr", "autocorr_delta" + curr_d + "_dim_" + str(dim) + "_rep_" + str(i) + ".png"),
                          lag=lag)

            ### prHMC ###
            print("Simulating prHMC")
            extra_pars_prHMC = dict({
                "L_start": L_start,
                "L_noise": L_noise,
                "epsilon_noise": epsilon_noise,
                "epsilon_start": epsilon_start,
                "version": "uniform",
                "adapt_L": adapt_L,
                "quantile_ub": quantile_ub,
                "adapt_with_opt_epsilon": adapt_opt_epsilon,
                "refreshment_prob": refreshment_prob,
                "epsilon_bar": epsilon_bar,
                "gamma": gamma,
                "t0": t0,
                "kappa": kappa,
                "delta": d,
                "adapt_epsilon": adapt_epsilon
            })
            prHMC_test = prHMC(U, grad_U, start_position, extra_pars_prHMC, iter, dual_averaging)
            res_prHMC = prHMC_test.simulate()
            res_prHMC = res_prHMC[start:, :]

            # Compute metrics
            esjd_l_prhmc[i, :] = np.array((ESJD(res_prHMC) / prHMC_test.grad_evals, d))
            ess_s_prhmc[i, :] = [ESS(res_prHMC[:, i], 100) / prHMC_test.elapsed_time for i in
                                 range(res_prHMC.shape[1])] + [d]
            ess_l_prhmc[i, :] = [ESS(res_prHMC[:, i], 100) / prHMC_test.grad_evals for i in
                                 range(res_prHMC.shape[1])] + [d]

            # Plot autocorrelation for one of the dimensions
            plot_autocorr(res_prHMC, col,
                          os.path.join("Plots", "prHMC", name, "autocorr", "autocorr_delta_" + curr_d + "_dim_" + str(dim) + "_rep_" + str(i) + ".png"),
                          lag=lag)

            # NUTS
            print("Simulating NUTS")
            extra_pars_NUTS = dict({
                "epsilon_bar": epsilon_bar,
                "gamma": gamma,
                "t0": t0,
                "kappa": kappa,
                "delta": d,
                "adapt_epsilon": adapt_epsilon,
                "delta_max": delta_max,
                "start_epsilon": epsilon_start
            })
            nuts_test = NUTS(U, grad_U, start_position, extra_pars_NUTS, iter)
            res_nuts = nuts_test.simulate()

            # Compute metrics
            esjd_l_nuts[i, :] = np.array((ESJD(res_nuts) / nuts_test.grad_evals, d))
            ess_s_nuts[i, :] = [ESS(res_nuts[:, i], 100) / nuts_test.elapsed_time for i in range(res_nuts.shape[1])] + [
                d]
            ess_l_nuts[i, :] = [ESS(res_nuts[:, i], 100) / nuts_test.grad_evals for i in range(res_nuts.shape[1])] + [d]

            # Plot autocorrelation for one of the dimensions
            plot_autocorr(res_nuts, col,
                          os.path.join("Plots", "NUTS", name, "autocorr", "autocorr_" + curr_d + "_" + str(i) + ".png"),
                          lag=lag)

        np.savez(os.path.join("Results", "HMC", name, "results_delta" + curr_d + "_dim_" + str(dim) + ".npz"), esjd_l_hmc, ess_s_hmc, ess_l_hmc)
        np.savez(os.path.join("Results", "wiggle_HMC", name,  "results_delta" + curr_d + "_dim_" + str(dim) + ".npz"), esjd_l_wiggle_hmc,
                 ess_s_wiggle_hmc, ess_l_wiggle_hmc)
        np.savez(os.path.join("Results", "eHMC", name, "results_delta" + curr_d + "_dim_" + str(dim) + ".npz"), esjd_l_ehmc, ess_s_ehmc, ess_l_ehmc)
        np.savez(os.path.join("Results", "prHMC", name, "results_delta" + curr_d + "_dim_" + str(dim) + ".npz"), esjd_l_prhmc, ess_s_prhmc, ess_l_prhmc)
        np.savez(os.path.join("Results", "NUTS", name, "results_delta" + curr_d + "_dim_" + str(dim) + ".npz"), esjd_l_nuts, ess_s_nuts, ess_l_nuts)

def set_extra_pars(extra_pars):
    L_start = extra_pars["L_start"]
    L_noise = extra_pars["L_noise"]
    epsilon_noise = extra_pars["epsilon_noise"]
    epsilon_start = extra_pars["epsilon_start"]
    epsilon_bar = extra_pars["epsilon_bar"]
    gamma = extra_pars["gamma"]
    t0 = extra_pars["t0"]
    kappa = extra_pars["kappa"]
    adapt_epsilon = extra_pars["adapt_epsilon"]
    threshold = extra_pars["threshold"]
    quantile_ub = extra_pars["quantile_ub"]
    quantile_lb = extra_pars["quantile_lb"]
    adapt_L = extra_pars["adapt_L"]
    adapt_opt_epsilon = extra_pars["adapt_opt_epsilon"]
    refreshment_prob = extra_pars["refreshment_prob"]
    dual_averaging = extra_pars["dual_averaging"]
    samp_random = extra_pars["samp_random"]
    delta_max = extra_pars["delta_max"]

    return L_start, L_noise, epsilon_noise, epsilon_start, epsilon_bar, gamma, t0, kappa, adapt_epsilon, \
        threshold, quantile_ub, quantile_lb, adapt_L, adapt_opt_epsilon, \
        refreshment_prob, delta_max, dual_averaging, samp_random