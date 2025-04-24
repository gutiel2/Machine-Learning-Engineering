# Project: Leveraging Machine Learning for Turbulence Modeling  
**Author:** Mateo Gutierrez

## Objective
Turbulence modeling in Computational Fluid Dynamics (CFD) can be extremely time- and resource-intensive when using high-fidelity approaches such as Direct Numerical Simulation (DNS) or Large Eddy Simulation (LES). The objective of this project is to explore the use of Reynolds-Averaged Navier-Stokes (RANS) simulations in conjunction with a Tensor Basis Neural Network (TBNN) to predict the Reynolds stress tensor. This approach aims to enhance the resolution and accuracy of RANS results while maintaining computational efficiency.

## Dataset
The dataset is sourced from Kaggle:  
[ML Turbulence Dataset – Kaggle](https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset/versions/3/data)

It includes a combination of RANS and DNS/LES data for 29 two-dimensional incompressible flow cases. Each case is stored in `.npy` format and includes flow variables such as velocity, pressure, turbulence kinetic energy, and Reynolds stresses.

##  Model Overview
The model used is a Tensor Basis Neural Network (TBNN), based on the formulation by Ling et al. (2016) and Pope (1975). It predicts the anisotropic part of the Reynolds stress tensor using physically invariant input features derived from the mean flow (strain-rate and rotation-rate tensors).

## Instructions to Run the Code
Required Dependencies: torch, numpy, matplotlib, and random.


1. **Choose a case:**  
   Set the target dataset and case. Default:  
   - Turbulence model: `komegasst`  
   - Case: `CNDV_12600`  

   DNS/LES labels for comparison are found in the `archive/labels` directory.

2. **Set model parameters:**  
   Default TBNN settings:
   - `hidden_dim = 256`  
   - `batch_size = 64`  
   - `learning_rate = 1e-5`  
   - `epochs = 200`  
   - `dropout = 0.1`

3. **Run the model:**  
   Execute the training script. It will train the TBNN on RANS input features and predict the Reynolds stress anisotropy tensor.

4. **Optional (for tuning):**  
   - `grid_search.py`: Performs randomized hyperparameter tuning  
   - `optuna_search.py`: Uses Optuna for Bayesian hyperparameter optimization  
   These scripts are not required for model execution but were used in the tuning phase.

## Summary of Results
The TBNN model showed strong agreement with DNS/LES results.  
- **Quantitative:**  
  - Mean Squared Error (MSE) and Mean Absolute Error (MAE) both converged to values near $10^{-3}$ after 200 epochs.
- **Qualitative:**  
  - Visualization of the predicted shear stress component $S_{12}$ closely matched DNS/LES data, capturing key turbulent structures more accurately than traditional RANS models.

The results validate that TBNNs can effectively bridge the gap between fast, low-fidelity RANS simulations and high-fidelity DNS/LES benchmarks.
