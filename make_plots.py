from plots import *

methods = ["HMC", "wiggle_HMC", "eHMC", "prHMC", "NUTS"]
models = ["mvn", "log_reg", "banana"]
dims = [2, 10, 20]
for model in models:
    if model == "banana":
        for d in dims:
            plot_method_2(methods, model, d)
    else:
        plot_method_2(methods, model)
