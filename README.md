# Fatigue Life Analysis using Basquin's Equation

This repository contains Python code to analyze fatigue life data using Basquin's equation. The code compares experimental data with predictions from a Hybrid Physics-Informed Neural Network (HPINN) and a synthetic dataset generated using Basquin's equation as a control. The goal is to evaluate the accuracy and performance of the HPINN model against experimental and synthetic data.

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Usage](#usage)
4. [Code Structure](#code-structure)
5. [Results](#results)
6. [License](#license)

---

## Introduction

Fatigue life analysis is critical for understanding material behavior under cyclic loading. Basquin's equation is widely used to model the relationship between stress amplitude and fatigue life. This project:
- Fits Basquin's equation to experimental data.
- Compares experimental results with predictions from a Hybrid Physics-Informed Neural Network (HPINN).
- Uses a synthetic dataset generated using Basquin's equation as a control for validation.

The code produces:
- A plot comparing experimental, HPINN, and synthetic Basquin predictions.
- Tabulated results including R² and RMSE values for each dataset.


## Requirements

To run this code, you need the following:

### Software
- Python 3.x

### Python Libraries
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

You can install the required libraries using the following command
pip install numpy pandas matplotlib scipy
