from plots import *

methods = ["HMC", "wiggle_HMC", "eHMC", "prHMC", "NUTS"]
for i in methods:
    plot_method(i)