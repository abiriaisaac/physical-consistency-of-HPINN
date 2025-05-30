# -*- coding: utf-8 -*-
"""
Script documentation header with creation date, institution, and author information
Created on Fri Mar 21 11:29:48 2025
Beijing Institute of Technology
@author: abiria Isaac
"""

# Import required libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
from scipy.optimize import curve_fit  # For curve fitting

# Set global plot parameters
plt.rcParams.update({
    'font.size': 16,  # Default font size
    'font.weight': 'bold',  # Bold font
    'font.family': 'serif'  # Serif font family
})

# Load data from Excel file (commented options show alternative files)
data = pd.read_excel('HPINN_Test_Results_synthetic_hpinn_shap.xlsx')  # Main data file

# Extract data columns into numpy arrays
stress = data['stress'].values  # Extract stress values
actual_cycles = data['experimental_cycles'].values  # Extract experimental cycle values
predicted_cycles_pinn = data['HPINN'].values  # Extract HPINN predicted cycles
fitted_cycles_basquin = data['fitted_cycles'].values  # Extract Basquin fitted cycles

# Define Basquin's law function
def basquin_func(Nf, sigma_f_prime, b):
    return sigma_f_prime * (Nf ** b)  # Basquin's equation

# Set initial parameters for curve fitting
initial_guess = [1e3, -1]  # Initial values for sigma_f_prime and b

# Perform curve fitting for actual experimental data
params_actual, _ = curve_fit(basquin_func, actual_cycles, stress, p0=initial_guess)
sigma_f_prime_actual, b_actual = params_actual  # Unpack fitted parameters

# Perform curve fitting for HPINN predicted data
params_pinn, _ = curve_fit(basquin_func, predicted_cycles_pinn, stress, p0=initial_guess)
sigma_f_prime_pinn, b_pinn = params_pinn  # Unpack fitted parameters

# Perform curve fitting for Basquin fitted data
params_basquin, _ = curve_fit(basquin_func, fitted_cycles_basquin, stress, p0=initial_guess)
sigma_f_prime_basquin, b_basquin = params_basquin  # Unpack fitted parameters

# Create range of cycles for smooth curve plotting
Nf_range = np.linspace(min(np.concatenate([actual_cycles, predicted_cycles_pinn, fitted_cycles_basquin])), 
                       max(np.concatenate([actual_cycles, predicted_cycles_pinn, fitted_cycles_basquin])), 500)

# Calculate fitted curves using obtained parameters
fitted_actual = basquin_func(Nf_range, sigma_f_prime_actual, b_actual)
fitted_pinn = basquin_func(Nf_range, sigma_f_prime_pinn, b_pinn)
fitted_basquin = basquin_func(Nf_range, sigma_f_prime_basquin, b_basquin)

# Define R-squared calculation function
def calculate_r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)  # Residual sum of squares
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)  # R-squared formula

# Calculate R-squared values for each fit
r2_actual = calculate_r2(stress, basquin_func(actual_cycles, sigma_f_prime_actual, b_actual))
r2_pinn = calculate_r2(stress, basquin_func(predicted_cycles_pinn, sigma_f_prime_pinn, b_pinn))
r2_basquin = calculate_r2(stress, basquin_func(fitted_cycles_basquin, sigma_f_prime_basquin, b_basquin))

# Create figure for plotting
plt.figure(figsize=(12, 8))

# Plot scatter points for each dataset
plt.scatter(actual_cycles, stress, label='Experimental', color='blue', marker='o', s=50)
plt.scatter(predicted_cycles_pinn, stress, label='HPINN', color='green', marker='s', s=50)
plt.scatter(fitted_cycles_basquin, stress, label='Synthetic Basquin', color='red', marker='^', s=50)

# Plot fitted curves with equation and R-squared values
plt.plot(Nf_range, fitted_actual, label=f'Experimental \n$\\sigma_a = {sigma_f_prime_actual:.2e} N^{{{b_actual:.2f}}}$\n$R^2 = {r2_actual:.2f}$', color='blue', linestyle='--')
plt.plot(Nf_range, fitted_pinn, label=f'HPINN\n$\\sigma_a = {sigma_f_prime_pinn:.2e}N^{{{b_pinn:.2f}}}$\n$R^2 = {r2_pinn:.2f}$', color='green', linestyle='--')
plt.plot(Nf_range, fitted_basquin, label=f'Synthetic Basquin\n$\\sigma_a =  {sigma_f_prime_basquin:.2e}N^{{{b_basquin:.2f}}}$\n$R^2 = {r2_basquin:.2f}$', color='red', linestyle='--')

# Set axis limits (currently using DATA1 chapter5 settings)
xmin = 4e4  # Minimum x-axis value
xmax = 1.1e6  # Maximum x-axis value
ymin = 100  # Minimum y-axis value
ymax = 190  # Maximum y-axis value

# Apply axis limits
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# Set plot labels and features
plt.xlabel('Fatigue Life (N)', fontsize=16, fontweight='bold')
plt.ylabel('Stress Amplitude ($\\sigma_a$)', fontsize=16, fontweight='bold')
plt.legend()  # Show legend
plt.grid(True, linestyle='--', linewidth=2)  # Add grid lines

# Customize plot borders
ax = plt.gca()  # Get current axes
for spine in ax.spines.values():
    spine.set_linewidth(2)  # Set border thickness

# Display the plot
plt.show()

# Define RMSE calculation function
def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))  # RMSE formula

# Calculate RMSE for each dataset
rmse_actual = calculate_rmse(stress, basquin_func(actual_cycles, sigma_f_prime_actual, b_actual))
rmse_pinn = calculate_rmse(stress, basquin_func(predicted_cycles_pinn, sigma_f_prime_pinn, b_pinn))
rmse_basquin = calculate_rmse(stress, basquin_func(fitted_cycles_basquin, sigma_f_prime_basquin, b_basquin))

# Create results table
results = pd.DataFrame({
    'Dataset': ['Actual', 'HPINN Predicted', 'Synthetic Basquin'],
    'a_basquin': [sigma_f_prime_actual, sigma_f_prime_pinn, sigma_f_prime_basquin],
    'b_basquin': [b_actual, b_pinn, b_basquin],
    'R2': [r2_actual, r2_pinn, r2_basquin],
    'RMSE': [rmse_actual, rmse_pinn, rmse_basquin]
})

# Print results
print(results)

# Save results to Excel file
results.to_excel('Basquin_Fit_Results.xlsx', index=False)
