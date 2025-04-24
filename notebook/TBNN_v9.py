import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary
import matplotlib.pyplot as plt
import random


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

dataset = 'komegasst'
case1 = ['CNDV_12600']

# ---------------------------------------------------------------------
# 2. Loading Data from Dataset
# ---------------------------------------------------------------------

# RANS Parameters
gradU   = loadCombinedArray(case1, 'gradU')
k   = loadCombinedArray(case1, 'k')
x   = loadCombinedArray(case1, 'Cx')
y   = loadCombinedArray(case1, 'Cy')

# DNS/LES Parameters
tau     = loadTruthArray(case1, 'tau')  # shape (N, 3, 3)

# ---------------------------------------------------------------------
# 3. Computing the Strain Rate & Rotation Rate
# ---------------------------------------------------------------------
gradU_T = np.transpose(gradU, (0, 2, 1))
strain_rate   = 0.5 * (gradU + gradU_T)
rotation_rate = 0.5 * (gradU - gradU_T)
N = strain_rate.shape[0]

# ---------------------------------------------------------------------
# 4. Defining TBNN with Dropout
# ---------------------------------------------------------------------
class TBNN(nn.Module):
    """
    Computes the tensor basis T_ij^(n) using strain-rate and rotation-rate tensors
    based on the ten symmetric, solenoidal tensors defined in Pope (1975).
    """
    def __init__(self, input_dim, hidden_dim, num_coeffs, output_dim, dropout_prob=0.0):
        super(TBNN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),
        )
        self.coeff_layer = nn.Linear(hidden_dim, num_coeffs)
        self.output_dim = output_dim

    def forward(self, x, tensor_basis):
        """
        Inputs:
            x (Tensor): Scalar invariants of shape (batch_size, input_dim)
            tensor_basis (Tensor): Tensor basis of shape (batch_size, 10, 3, 3)
        Outputs:
            a_pred (Tensor): Anisotropic part of Reynolds stress (batch_size, 3, 3)
        """
        out = self.hidden(x)
        coeffs = self.coeff_layer(out)  # (batch_size, num_coeffs)
        coeffs = coeffs.unsqueeze(-1).unsqueeze(-1)  # (batch_size, num_coeffs, 1, 1)
        a_pred = torch.sum(coeffs * tensor_basis, dim=1)  # (batch_size, 3, 3)
        return a_pred

# ---------------------------------------------------------------------
# 5. Compute Tensor Basis
# ---------------------------------------------------------------------
def compute_tensor_basis(S, Omega):
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

T_basis = compute_tensor_basis(strain_rate, rotation_rate)
tensor_basis_torch = torch.tensor(T_basis, dtype=torch.float32)  # (N, 10, 3, 3)

# ---------------------------------------------------------------------
# 6. Prepare Input Features
# ---------------------------------------------------------------------
def compute_lambda(strain_rate, rotation_rate):
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

I1, I2, I3, I4, I5 = compute_lambda(strain_rate, rotation_rate)
input_features = np.stack([I1, I2, I3, I4, I5], axis=1)  # (N, 5)

# Scaling input features
features_mean = np.mean(input_features, axis=0)
features_std = np.std(input_features, axis=0) + 1e-20
input_features_scaled = (input_features - features_mean) / features_std
features_tensor = torch.tensor(input_features_scaled, dtype=torch.float32)

# Keep tau unscaled
actual_R_tensor = torch.tensor(tau, dtype=torch.float32)  # (N, 3, 3)
k_tensor = torch.tensor(k, dtype=torch.float32)           # (N,)

# ---------------------------------------------------------------------
# 7. Custom Dataset to Ensure Alignment
# ---------------------------------------------------------------------
class TBNNDataset(Dataset):

    def __init__(self, features, tensor_basis, R_target, k_values):
        self.features = features
        self.tensor_basis = tensor_basis
        self.R_target = R_target
        self.k_values = k_values
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return (self.features[idx],
                self.tensor_basis[idx],
                self.R_target[idx],
                self.k_values[idx])

dataset_full = TBNNDataset(features_tensor, tensor_basis_torch, actual_R_tensor, k_tensor)

# ---------------------------------------------------------------------
# 8. Split into Train/Val, DataLoaders
# ---------------------------------------------------------------------

train_size = int(0.8 * N)
val_size = N - train_size
train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------------------------------------------------
# 9. Instantiate Model, Optimizer, Loss
# ---------------------------------------------------------------------
input_dim = 5
hidden_dim = 256
num_coeffs = 10
output_dim = 3

model = TBNN(input_dim, hidden_dim, num_coeffs, output_dim, dropout_prob=0.1)

# Show a quick summary
sample_feats, sample_basis, _, _ = train_dataset[:10]
summary(model, input_data=(sample_feats.unsqueeze(0), sample_basis.unsqueeze(0)))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# ---------------------------------------------------------------------
# 10. Training & Validation Loop with MSE and MAE Metrics
# ---------------------------------------------------------------------
num_epochs = 200
train_mse_losses = []
val_mse_losses = []
train_mae_losses = []
val_mae_losses = []
val_r2_scores = []

for epoch in range(num_epochs):
    model.train()
    running_mse = 0.0
    running_mae = 0.0
    for batch_feats, batch_basis, batch_tau, batch_k in train_loader:
        optimizer.zero_grad()
        # Forward: predict anisotropy a_pred (batch_size, 3, 3)
        a_pred = model(batch_feats, batch_basis)
        
        # Reconstruct τ_pred = 2 * k * (I/3 + a_pred)
        batch_size_current = a_pred.size(0)
        batch_k_expanded = batch_k.unsqueeze(-1).unsqueeze(-1)  # (batch_size,1,1)
        I_mat = torch.eye(output_dim, device=a_pred.device).unsqueeze(0).expand(batch_size_current, -1, -1)
        isotropic_part = (1/3.) * I_mat
        tau_pred = 2.0 * batch_k_expanded * (isotropic_part + a_pred)
        
        mse_loss = criterion(tau_pred, batch_tau)
        mae_loss = torch.mean(torch.abs(tau_pred - batch_tau))
        
        mse_loss.backward()
        optimizer.step()
        
        running_mse += mse_loss.item() * batch_size_current
        running_mae += mae_loss.item() * batch_size_current
    
    epoch_train_mse = running_mse / len(train_loader.dataset)
    epoch_train_mae = running_mae / len(train_loader.dataset)
    train_mse_losses.append(epoch_train_mse)
    train_mae_losses.append(epoch_train_mae)
    
    # Validation
    model.eval()
    val_mse = 0.0
    val_mae = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        """
        This block performs evaluation of the trained TBNN model on the validation set.
        Disables gradient computation to reduce memory usage and speed up inference.
        Reconstructs the full Reynolds stress tensor τ_ij from predicted anisotropy a_ij and turbulent kinetic energy k.
        Computes validation loss metrics (MSE and MAE) for each batch.
        """
        for batch_feats, batch_basis, batch_tau, batch_k in val_loader:
            a_pred_val = model(batch_feats, batch_basis)
            bs_val = a_pred_val.size(0)
            batch_k_val = batch_k.unsqueeze(-1).unsqueeze(-1)
            I_mat_val = torch.eye(output_dim, device=a_pred_val.device).unsqueeze(0).expand(bs_val, -1, -1)
            iso_part_val = (1/3.) * I_mat_val
            tau_pred_val = 2.0 * batch_k_val * (iso_part_val + a_pred_val)
            
            mse_val = criterion(tau_pred_val, batch_tau)
            mae_val = torch.mean(torch.abs(tau_pred_val - batch_tau))
            
            val_mse += mse_val.item() * bs_val
            val_mae += mae_val.item() * bs_val
            
            all_preds.append(tau_pred_val)
            all_labels.append(batch_tau)
    
    epoch_val_mse = val_mse / len(val_loader.dataset)
    epoch_val_mae = val_mae / len(val_loader.dataset)
    val_mse_losses.append(epoch_val_mse)
    val_mae_losses.append(epoch_val_mae)
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train MSE: {epoch_train_mse:.12g} | Train MAE: {epoch_train_mae:.12g} | "
          f"Val MSE: {epoch_val_mse:.12g} | Val MAE: {epoch_val_mae:.12g} | "
        )

# ---------------------------------------------------------------------
# 11. Plot Training & Validation Metrics
# ---------------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_mse_losses, label='Train MSE')
plt.plot(range(1, num_epochs+1), val_mse_losses, label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_mae_losses, label='Train MAE')
plt.plot(range(1, num_epochs+1), val_mae_losses, label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.title('Training and Validation MAE Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), val_r2_scores, label='Validation R^2')
plt.xlabel('Epoch')
plt.ylabel('R^2')
plt.title('Validation R^2 Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

model.eval()

true_vals = []
pred_vals = []

# Lists to store all flattened true and predicted values
all_true_vals = []
all_pred_vals = []

# Iterate over a set of sample indices or the entire validation set
for idx in range(len(val_dataset)):
    feats, basis, true_tau, kk = val_dataset[idx]
    feats = feats.unsqueeze(0)   # (1, 5)
    basis = basis.unsqueeze(0)   # (1, 10, 3, 3)
    with torch.no_grad():
        a_pred_sample = model(feats, basis)
    I_sample = torch.eye(output_dim).unsqueeze(0)  # (1, 3, 3)
    tau_pred_sample = 2.0 * kk.item() * (I_sample/3.0 + a_pred_sample)
    
    # Flatten the 3x3 tensor into a 1D array and append
    all_true_vals.append(true_tau.numpy().flatten())
    all_pred_vals.append(tau_pred_sample.numpy().flatten())

# Concatenate the lists into single arrays
all_true_vals = np.concatenate(all_true_vals)
all_pred_vals = np.concatenate(all_pred_vals)

def rmse(pred_tensor, true_tensor):
    """
    Calculate the RMSE for the lower triangular part of the tensor.
    The RMSE is calculated over the lower triangular elements of the tensor, excluding the diagonal.

    Inputs:
    pred_tensor: torch.Tensor, predicted tensor of shape (N, 3, 3)
    true_tensor: torch.Tensor, true tensor of shape (N, 3, 3)

    Outputs:
    rmse: float, root mean square error for the lower triangular part of the tensor
    """


    N = pred_tensor.shape[0]  # Number of samples (N_data)

    # Boolean mask to select lower triangle: i >= j
    mask = torch.tensor([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]
    ], dtype=torch.bool)

    # Element-wise squared error
    error = (pred_tensor - true_tensor) ** 2

    # Apply the mask: only get lower triangular elements (6 per sample)
    lower_tri_error = error[:, mask]  # shape: (N, 6)

    # Mean over all 6 elements and N samples
    mse = torch.sum(lower_tri_error) / (6 * N)
    return torch.sqrt(mse).item()

# Stack and reshape your prediction and true values
pred_tensor = torch.tensor(np.stack(all_pred_vals)).reshape(-1, 3, 3)
true_tensor = torch.tensor(np.stack(all_true_vals)).reshape(-1, 3, 3)

# Compute and print RMSE
rmse_value = rmse(pred_tensor, true_tensor)

print(f"RMSE (lower triangle): {rmse_value:.6f}")

all_tau_pred = []

sample_indices = range(len(val_dataset))
for idx in sample_indices:
    feats, basis, true_tau, kk = val_dataset[idx]
    feats = feats.unsqueeze(0)   # (1, 5)
    basis = basis.unsqueeze(0)   # (1, 10, 3, 3)
    with torch.no_grad():
        a_pred_sample = model(feats, basis)
    I_sample = torch.eye(output_dim).unsqueeze(0)
    tau_pred_sample = 2.0 * kk.item() * (I_sample / 3.0 + a_pred_sample)
    
    all_tau_pred.append(tau_pred_sample.squeeze(0))  # Remove batch dim

# Stack all into a single tensor: shape (N_val, 3, 3)
tau_pred_tensor = torch.stack(all_tau_pred, dim=0)

# Compute Frobenius norm for each sample
tau_norm = torch.linalg.norm(tau_pred_tensor, dim=(1, 2)).numpy()
# Extract validation coordinates only
x_val = x[val_dataset.indices]
y_val = y[val_dataset.indices]

import matplotlib.colors as mcolors
# Plotting the shear stress component τ_12

plt.scatter(x_val, y_val, c=tau_norm, cmap='turbo',
            norm=mcolors.LogNorm(vmin=10**-5, vmax=10**-1))
plt.xlabel('X')
plt.ylabel('Y')
plt.title(r'$\tau_{12}$ Shear Stress Component')
plt.colorbar(label=r'$\tau_{12}$')
plt.axis('equal')
plt.show()

