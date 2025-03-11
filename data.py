import yaml
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

#############################
# 1) Load data from YAML
#############################
config_str = """
default_glucose: [1.2, 2.4, 3.6, 4.8]
mu_37: [0.0274, 0.0235, 0.0362, 0.041]
mu_33: [0.011, 0.022, 0.032, 0.035]
"""
config = yaml.safe_load(config_str)

DEFAULT_GLUCOSE = np.array(config["default_glucose"])
MU_37 = np.array(config["mu_37"])
MU_33 = np.array(config["mu_33"])

#############################
# 2) Define & fit the model
#############################
def log_func(x, a, b):
    """Logarithmic growth model: mu = a * ln(x) + b."""
    return a * np.log(x) + b

params_37, _ = curve_fit(log_func, DEFAULT_GLUCOSE, MU_37)
params_33, _ = curve_fit(log_func, DEFAULT_GLUCOSE, MU_33)

#############################
# 3) Generate data
#############################
# For example, assume:
initial_count = 1.0e6  # 1 million cells initially
days = 1.0             # 1 day of culture
hours = days * 24.0    # convert days to hours

# Pick a range of glucose values to evaluate:
glucose_range = np.arange(0.5, 5.1, 0.1)

# Prepare lists to hold results
temp_list = []
glucose_list = []
viable_count_list = []

for T, params in zip([33, 37], [params_33, params_37]):
    for g in glucose_range:
        mu = log_func(g, *params)  # predicted specific growth rate
        # Final viable cell count after 'hours' hours:
        final_cell_count = initial_count * np.exp(mu * hours)

        temp_list.append(T)
        glucose_list.append(g)
        viable_count_list.append(final_cell_count)

#############################
# 4) Build the DataFrame
#############################
df = pd.DataFrame({
    "temperature": temp_list,
    "glucose": glucose_list,
    "viable_cell_density_count": viable_count_list
})

# Show the DataFrame in the console
# print(df)

# (Optional) save to CSV
df.to_csv("growth_data.csv", index=False)
