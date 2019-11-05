import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_method(name):
    mult = 1000
    esjds, ess_s, ess_l = load_data(name)
    #plots minimum ess_s, ess_l by delta
    min_ess_s = np.concatenate((np.min(ess_s[:, :-1], axis=1).reshape(-1, 1), ess_s[:, -1].reshape(-1, 1)), axis=1)
    min_ess_l = np.concatenate((np.min(ess_l[:, :-1], axis=1).reshape(-1, 1), ess_l[:, -1].reshape(-1, 1)), axis=1)

    mean_min_ess_s = compute_mean_esjds(min_ess_s)
    mean_min_ess_l = compute_mean_esjds(min_ess_l)

    plt.scatter(min_ess_s[:, 1], min_ess_s[:, 0])
    plt.plot(mean_min_ess_s[:, 0], mean_min_ess_s[:, 1])
    plt.xlabel("Delta")
    plt.ylabel("ESS/S")
    plt.title(name)
    plt.savefig(os.path.join("Plots", name, "ess_s.png"))
    plt.close()

    plt.scatter(min_ess_l[:, 1], mult*min_ess_l[:, 0])
    plt.plot(mean_min_ess_l[:, 0], mult*mean_min_ess_l[:, 1])
    plt.xlabel("Delta")
    plt.ylabel("ESS/L")
    plt.title(name)
    plt.savefig(os.path.join("Plots", name, "ess_l.png"))
    plt.close()

    mean_esjds = compute_mean_esjds(esjds)
    plt.scatter(esjds[:, 1], mult*esjds[:, 0])
    plt.plot(mean_esjds[:, 0], mult*mean_esjds[:, 1])
    plt.xlabel("Delta")
    plt.ylabel("ESJD/L")
    plt.title(name)
    plt.savefig(os.path.join("Plots", name, "esjd.png"))
    plt.close()

def load_data(name):
    files = os.listdir(os.path.join("Results", name))

    file_0 = files[0]
    dt = np.load(os.path.join("Results", name, file_0))
    esjds = dt["arr_0"]
    ess_s = dt["arr_1"]
    ess_l = dt["arr_2"]

    for f in files[1:]:
        dt = np.load(os.path.join("Results", name, f))
        esjds = np.concatenate((esjds, dt["arr_0"]))
        ess_s = np.concatenate((ess_s, dt["arr_1"]))
        ess_l = np.concatenate((ess_l, dt["arr_2"]))

    return esjds, ess_s, ess_l

def compute_mean_esjds(esjds):
    pd_df = pd.DataFrame(esjds, columns=["esjd", "deltas"])
    pd_df = np.array(pd_df.groupby("deltas", as_index=False).mean())
    return pd_df