# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:29:48 2025
Beijing Institute of Technology
@author: abiria Isaac
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({
    'font.size': 16,
    'font.weight': 'bold',
    'font.family': 'serif'
})

# Load the data (replace with your actual file path)


#data = pd.read_excel('HPINN_Test_Results_synthetic_Aluminium.xlsx')  # Adjust to your file location
#data = pd.read_excel('HPINN_Test_Results_synthetic_Titanium.xlsx')  # Adjust to your file location
data = pd.read_excel('HPINN_Test_Results_synthetic_chapter5.xlsx')  # Adjust to your file location
stress = data['stress'].values                      # Applied stress (σ)
actual_cycles = data['experimental_cycles'].values        # Actual fatigue life (Nf)
predicted_cycles_pinn = data['HPINN'].values  # Predicted fatigue life (PINN)
fitted_cycles_basquin = data['fitted_cycles'].values  # Fitted Basquin fatigue life






# Define the Basquin function
def basquin_func(Nf, sigma_f_prime, b):
    return sigma_f_prime * (Nf ** b)

# Fit the Basquin function to each dataset
initial_guess = [1e3, -1]  # Initial guess for [sigma_f_prime, b]

# Fit for Actual Cycles
params_actual, _ = curve_fit(basquin_func, actual_cycles, stress, p0=initial_guess)
sigma_f_prime_actual, b_actual = params_actual

# Fit for Predicted Cycles (PINN)
params_pinn, _ = curve_fit(basquin_func, predicted_cycles_pinn, stress, p0=initial_guess)
sigma_f_prime_pinn, b_pinn = params_pinn

# Fit for Fitted Cycles (Basquin)
params_basquin, _ = curve_fit(basquin_func, fitted_cycles_basquin, stress, p0=initial_guess)
sigma_f_prime_basquin, b_basquin = params_basquin

# Generate Nf range for smooth curve plotting (broad range to ensure all data is captured)
Nf_range = np.linspace(min(np.concatenate([actual_cycles, predicted_cycles_pinn, fitted_cycles_basquin])), 
                       max(np.concatenate([actual_cycles, predicted_cycles_pinn, fitted_cycles_basquin])), 500)

# Calculate fitted curves using the Basquin function
fitted_actual = basquin_func(Nf_range, sigma_f_prime_actual, b_actual)
fitted_pinn = basquin_func(Nf_range, sigma_f_prime_pinn, b_pinn)
fitted_basquin = basquin_func(Nf_range, sigma_f_prime_basquin, b_basquin)

# Function to calculate R² value
def calculate_r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot)

# Calculate R² values for each fitted curve
r2_actual = calculate_r2(stress, basquin_func(actual_cycles, sigma_f_prime_actual, b_actual))
r2_pinn = calculate_r2(stress, basquin_func(predicted_cycles_pinn, sigma_f_prime_pinn, b_pinn))
r2_basquin = calculate_r2(stress, basquin_func(fitted_cycles_basquin, sigma_f_prime_basquin, b_basquin))

# Plot Stress vs Actual, Predicted (PINN), and Fitted (Basquin)
plt.figure(figsize=(12, 8))

# Scatter plots (for Actual Cycles, Predicted Cycles (PINN), and Fitted Cycles (Basquin))
plt.scatter(actual_cycles, stress, label='Experimental', color='blue', marker='o', s=50)
plt.scatter(predicted_cycles_pinn, stress, label='HPINN', color='green', marker='s', s=50)
plt.scatter(fitted_cycles_basquin, stress, label='Synthetic Basquin', color='red', marker='^', s=50)


# Fitted curves
plt.plot(Nf_range, fitted_actual, label=f'Experimental \n$\\sigma_a = {sigma_f_prime_actual:.2e} N^{{{b_actual:.2f}}}$\n$R^2 = {r2_actual:.2f}$', color='blue', linestyle='--')
plt.plot(Nf_range, fitted_pinn, label=f'HPINN\n$\\sigma_a = {sigma_f_prime_pinn:.2e}N^{{{b_pinn:.2f}}}$\n$R^2 = {r2_pinn:.2f}$', color='green', linestyle='--')
plt
plt.plot(Nf_range, fitted_basquin, label=f'Synthetic Basquin\n$\\sigma_a =  {sigma_f_prime_basquin:.2e}N^{{{b_basquin:.2f}}}$\n$R^2 = {r2_basquin:.2f}$', color='red', linestyle='--')
plt




#LIMITS FOR DATA1(chapter5)
# Set manual limits for the x and y axes
xmin = 4e4  # Set the minimum x-axis value
xmax = 1.1e6  # Set the maximum x-axis value
ymin = 100   # Set the minimum y-axis value
ymax = 190  # Set the maximum y-axis value


#LIMITS FOR DATA1(Titanium)
 #Set manual limits for the x and y axes
#xmin = 1.0e7  # Set the minimum x-axis value
#xmax = 7.5e8  # Set the maximum x-axis value
#ymin = 150   # Set the minimum y-axis value
#ymax = 400  # Set the maximum y-axis value

#LIMITS FOR DATA4(Aluminium)
# Set manual limits for the x and y axes
#xmin = 7.4e4  # Set the minimum x-axis value
#xmax = 2.6e6  # Set the maximum x-axis value
#ymin = 110   # Set the minimum y-axis value
#ymax = 210  # Set the maximum y-axis value



# Apply the manual axis limits
plt.xlim(xmin, xmax)  # Set the x-axis limits
plt.ylim(ymin, ymax)  # Set the y-axis limits

# Set labels, title, and grid
plt.xlabel('Fatigue Life (N)', fontsize=16, fontweight='bold')  # Thicker x-axis label
plt.ylabel('Stress Amplitude ($\\sigma_a$)', fontsize=16, fontweight='bold')  # Thicker y-axis label
plt.legend()
plt.grid(True, linestyle='--', linewidth=2)







# Make the plot borders (spines) thicker
ax = plt.gca()  # Get the current axes
for spine in ax.spines.values():
    spine.set_linewidth(2)  # Set the border thickness

# Show the plot
plt.show()

# Function to calculate RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

# Calculate RMSE for each dataset
rmse_actual = calculate_rmse(stress, basquin_func(actual_cycles, sigma_f_prime_actual, b_actual))
rmse_pinn = calculate_rmse(stress, basquin_func(predicted_cycles_pinn, sigma_f_prime_pinn, b_pinn))
rmse_basquin = calculate_rmse(stress, basquin_func(fitted_cycles_basquin, sigma_f_prime_basquin, b_basquin))

# Tabulate results
results = pd.DataFrame({
    'Dataset': ['Actual', 'HPINN Predicted', 'Synthetic Basquin'],
    'a_basquin': [sigma_f_prime_actual, sigma_f_prime_pinn, sigma_f_prime_basquin],
    'b_basquin': [b_actual, b_pinn, b_basquin],
    'R2': [r2_actual, r2_pinn, r2_basquin],
    'RMSE': [rmse_actual, rmse_pinn, rmse_basquin]
})

# Display the results
print(results)

# Optional: Save the results to an Excel file
results.to_excel('Basquin_Fit_Results.xlsx', index=False)
