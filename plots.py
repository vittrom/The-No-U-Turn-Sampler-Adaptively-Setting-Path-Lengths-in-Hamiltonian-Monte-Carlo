import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

def plot_method(name, model, dim=None):
    mult = 1000
    esjds, ess_s, ess_l = load_data(name, model, dim=dim)
    #plots minimum ess_s, ess_l by delta
    min_ess_s = np.concatenate((np.min(ess_s[:, :-1], axis=1).reshape(-1, 1), ess_s[:, -1].reshape(-1, 1)), axis=1)
    min_ess_l = np.concatenate((np.min(ess_l[:, :-1], axis=1).reshape(-1, 1), ess_l[:, -1].reshape(-1, 1)), axis=1)

    mean_min_ess_s = compute_mean_esjds(min_ess_s)
    mean_min_ess_l = compute_mean_esjds(min_ess_l)

    plt.scatter(min_ess_s[:, 1], min_ess_s[:, 0])
    plt.plot(mean_min_ess_s[:, 0], mean_min_ess_s[:, 1])
    plt.xlabel("Target acceptance probability")
    plt.ylabel("ESS/S")
    plt.title(name)
    if dim is not None:
        filename = "dim_" + str(dim) + "_ess_s.pdf"
    else:
        filename = "ess_s.pdf"
    plt.savefig(os.path.join("Plots", name, model, filename))
    plt.close()

    plt.scatter(min_ess_l[:, 1], mult*min_ess_l[:, 0])
    plt.plot(mean_min_ess_l[:, 0], mult*mean_min_ess_l[:, 1])
    plt.xlabel("Target acceptance probability")
    plt.ylabel("ESS/L")
    plt.title(name)
    if dim is not None:
        filename = "dim_" + str(dim) + "_ess_l.pdf"
    else:
        filename = "ess_l.pdf"
    plt.savefig(os.path.join("Plots", name, model, filename))
    plt.close()

    mean_esjds = compute_mean_esjds(esjds)
    plt.scatter(esjds[:, 1], mult*esjds[:, 0])
    plt.plot(mean_esjds[:, 0], mult*mean_esjds[:, 1])
    plt.xlabel("Target acceptance probability")
    plt.ylabel("ESJD/L")
    plt.title(name)
    if dim is not None:
        filename = "dim_" + str(dim) + "_esjd.pdf"
    else:
        filename = "esjd.pdf"
    plt.savefig(os.path.join("Plots", name, model, filename))
    plt.close()

def load_data(name, model, dim=None):
    if model != "banana":
        files = os.listdir(os.path.join("Results", name, model))

        file_0 = files[0]
        dt = np.load(os.path.join("Results", name, model, file_0))
        esjds = dt["arr_0"]
        ess_s = dt["arr_1"]
        ess_l = dt["arr_2"]

        for f in files[1:]:
            dt = np.load(os.path.join("Results", name, model, f))
            esjds = np.concatenate((esjds, dt["arr_0"]))
            ess_s = np.concatenate((ess_s, dt["arr_1"]))
            ess_l = np.concatenate((ess_l, dt["arr_2"]))
    else:
        files = os.listdir(os.path.join("Results", name, model, "dim_" + str(dim)))

        file_0 = files[0]
        dt = np.load(os.path.join("Results", name, model, "dim_" + str(dim), file_0))
        esjds = dt["arr_0"]
        ess_s = dt["arr_1"]
        ess_l = dt["arr_2"]

        for f in files[1:]:
            dt = np.load(os.path.join("Results", name, model, "dim_" + str(dim), f))
            esjds = np.concatenate((esjds, dt["arr_0"]))
            ess_s = np.concatenate((ess_s, dt["arr_1"]))
            ess_l = np.concatenate((ess_l, dt["arr_2"]))

    return esjds, ess_s, ess_l

def compute_mean_esjds(esjds):
    pd_df = pd.DataFrame(esjds, columns=["esjd", "deltas"])
    pd_df = np.array(pd_df.groupby("deltas", as_index=False).mean())
    return pd_df

def plot_autocorr(data, name, col, save_path, lag=100):
    f = plot_acf(data[:, col], lags=lag)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title(name)
    plt.savefig(save_path)
    plt.close()

def plot_method_2(names, model, dim=None):
    mult = 1000
    fig = plt.figure()
    ax = plt.subplot(111)
    for name in names:
        _, ess_s, _ = load_data(name, model, dim=dim)
        ess_s[ess_s < 0] = 99999
        #plots minimum ess_s, ess_l by delta
        min_ess_s = np.concatenate((np.min(ess_s[:, :-1], axis=1).reshape(-1, 1), ess_s[:, -1].reshape(-1, 1)), axis=1)

        mean_min_ess_s = compute_mean_esjds(min_ess_s)

        if name == "wiggle_HMC":
            lab = "NWHMC"
        else:
            lab = name
        ax.plot(mean_min_ess_s[:, 0], mean_min_ess_s[:, 1], label=lab)

    plt.xlabel("Target acceptance probability")
    plt.ylabel("ESS/S")
    if dim is not None:
        filename = "dim_" + str(dim) + "_ess_s.pdf"
    else:
        filename = "ess_s.pdf"

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.7, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.8), shadow=False, ncol=1)

    plt.savefig(os.path.join("Plots", model, filename))
    plt.close()

    fig = plt.figure()
    ax = plt.subplot(111)
    for name in names:
        _, _, ess_l = load_data(name, model, dim=dim)
        ess_l[ess_l <0] = 99999
        # plots minimum ess_s, ess_l by delta
        min_ess_l = np.concatenate((np.min(ess_l[:, :-1], axis=1).reshape(-1, 1), ess_l[:, -1].reshape(-1, 1)), axis=1)

        mean_min_ess_l = compute_mean_esjds(min_ess_l)

        if name == "wiggle_HMC":
            lab = "NWHMC"
        else:
            lab = name
        ax.plot(mean_min_ess_l[:, 0], mean_min_ess_l[:, 1], label=lab)

    plt.xlabel("Target acceptance probability")
    plt.ylabel("ESS/L")
    if dim is not None:
        filename = "dim_" + str(dim) + "_ess_l.pdf"
    else:
        filename = "ess_l.pdf"

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.7, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.8), shadow=False, ncol=1)

    plt.savefig(os.path.join("Plots", model, filename))
    plt.close()

    fig = plt.figure()
    ax = plt.subplot(111)
    for name in names:
        esjds, _, _ = load_data(name, model, dim=dim)
        mean_esjds = compute_mean_esjds(esjds)
        if model == "log_reg":
            mean_esjds = mean_esjds[mean_esjds[:, 0] != 0.95, :]

        if name == "wiggle_HMC":
            lab = "NWHMC"
        else:
            lab = name
        ax.plot(mean_esjds[:, 0], mult * mean_esjds[:, 1], label=lab)
    plt.xlabel("Target acceptance probability")
    plt.ylabel("ESJD/L")
    # plt.title(name)
    if dim is not None:
        filename = "dim_" + str(dim) + "_esjd.pdf"
    else:
        filename = "esjd.pdf"

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.7, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.3, 0.8), shadow=False, ncol=1)

    plt.savefig(os.path.join("Plots", model, filename))
    plt.close()