import os

methods = ["HMC", "NUTS", "wiggle_HMC", "eHMC", "prHMC"]
models = ["mvn", "sv", "banana", "log_reg"]

#Results
for method in methods:
    for model in models:
        dir_path = os.path.join("Results", method, model)
        for the_file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

#Plots
for method in methods:
    for model in models:
        dir_path = os.path.join("Plots", method, model, "autocorr")
        for the_file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)