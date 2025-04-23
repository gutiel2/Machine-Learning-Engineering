import matplotlib.pyplot as plt
import numpy as np


"""
File: preProcessing.py

Functions within this script:
    loadCombinedArray(cases, field)
    loadTruthArray(cases, field)
    computeInputs(gradU)
    computeLambda(strain_rate, rotation_rate)
    computeTensorBasis(S, Omega)

This script contains functions to load data from the RANS and DNS/LES datasets, compute the strain rate and rotation rate tensors,
 compute the lambda functions, and compute the tensor basis.

"""

# ---------------------------------------------------------------------
# 1. Data Loading Functions
# ---------------------------------------------------------------------
def loadCombinedArray(cases, field):
    """
    Combs through the RANS dataset and finds the corresponding case and field. It will then import the data and return it as a numpy array.

    Inputs:
    cases: list of strings, each string is a case name
    field: string, the field name to load (e.g., 'gradU', 'k', etc.)
    """
    data = np.concatenate([
        np.load(r'C:\Users\Mateo Gutierrez\OneDrive\Spring 2025\Machine Learning Engineering\Project\archive'
                + '\\' + dataset + '\\' + dataset + '_' + case + '_' + field + '.npy')
        for case in cases
    ])
    return data

def loadTruthArray(cases, field):
    """
    Combs through the DNS/LES dataset and finds the corresponding case and field. It will then import the data and return it as a numpy array.

    Inputs:
    cases: list of strings, each string is a case name
    field: string, the field name to load (e.g., 'gradU', 'k', etc.)
    """
    data = np.concatenate([
        np.load(r'C:\Users\Mateo Gutierrez\OneDrive\Spring 2025\Machine Learning Engineering\Project\archive\labels'
                + '\\' + case + '_' + field + '.npy')
        for case in cases
    ])
    return data

def computeInputs(gradU):
    """
    Computes the lambda functions for the strain rate and rotation rate tensors according to Pope (1975).

    Inputs:
    gradU: float, gradient of velocity tensor

    Outputs:
    strain_rate: float, computed strain rate tensor
    rotation_rate: float, computed rotation rate tensor
    """
    gradU_T = np.transpose(gradU, (0, 2, 1))
    strain_rate   = 0.5 * (gradU + gradU_T)
    rotation_rate = 0.5 * (gradU - gradU_T)
    N = strain_rate.shape[0]

    return strain_rate, rotation_rate, N

# ---------------------------------------------------------------------
# 6. Preparing Lambda Functions
# ---------------------------------------------------------------------
def computeLambda(strain_rate, rotation_rate):
    """
    Computes the lambda functions for the strain rate and rotation rate tensors according to Pope (1975).

    Inputs:
    strain_rate: float, computed strain rate tensor
    rotation_rate: float, computed rotation rate tensor

    Outputs:
    I1: float, first invariant of the strain rate tensor
    I2: float, second invariant of the strain rate tensor   
    I3: float, third invariant of the strain rate tensor
    I4: float, fourth invariant of the strain rate tensor
    I5: float, fifth invariant of the strain rate tensor
    """
    I1 = np.trace(np.matmul(strain_rate, strain_rate), axis1=1, axis2=2)
    I2 = np.trace(np.matmul(rotation_rate, rotation_rate), axis1=1, axis2=2)

    S2 = np.matmul(strain_rate, strain_rate)
    I3 = np.trace(np.matmul(S2, strain_rate), axis1=1, axis2=2)

    R2 = np.matmul(rotation_rate, rotation_rate)
    I4 = np.trace(np.matmul(R2, strain_rate), axis1=1, axis2=2)

    I5 = np.trace(np.matmul(R2, S2), axis1=1, axis2=2)
    return I1, I2, I3, I4, I5


# ---------------------------------------------------------------------
# 5. Compute Tensor Basis
# ---------------------------------------------------------------------
def computeTensorBasis(S, Omega):
    """
    Computes the tensor basis from the strain rate and rotation rate tensors according to Pope (1975).

    Inputs:
    S: float, strain rate tensor
    Omega: float, rotation rate tensor

    Outputs:
    T: float, tensor basis
    """
    N = S.shape[0]
    I = np.eye(3)[None, :, :]
    T = np.zeros((N, 10, 3, 3))
    
    T[:, 0] = S
    T[:, 1] = np.matmul(S, Omega) - np.matmul(Omega, S)
    S2 = np.matmul(S, S)
    trace_S2 = np.trace(S2, axis1=1, axis2=2)[:, None, None]
    T[:, 2] = S2 - (1/3.) * I * trace_S2
    
    Omega2 = np.matmul(Omega, Omega)
    trace_Omega2 = np.trace(Omega2, axis1=1, axis2=2)[:, None, None]
    T[:, 3] = Omega2 - (1/3.) * I * trace_Omega2
    
    T[:, 4] = np.matmul(Omega, S2) - np.matmul(S2, Omega)
    SOmega2 = np.matmul(S, Omega2)
    trace_SOmega2 = np.trace(SOmega2, axis1=1, axis2=2)[:, None, None]
    T[:, 5] = np.matmul(Omega2, S) + np.matmul(S, Omega2) - (2/3.) * I * trace_SOmega2
    
    T[:, 6] = np.matmul(np.matmul(Omega, S), Omega2) - np.matmul(np.matmul(Omega2, S), Omega)
    T[:, 7] = np.matmul(np.matmul(S, Omega), S2) - np.matmul(np.matmul(S2, Omega), S)
    T[:, 8] = np.matmul(np.matmul(Omega, S2), Omega2) - np.matmul(np.matmul(Omega2, S2), Omega)
    T[:, 9] = np.matmul(np.matmul(S, Omega2), S2) - np.matmul(np.matmul(S2, Omega2), S)
    return T