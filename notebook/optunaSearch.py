
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchinfo import summary
import optuna

# ---------------------------------------------------------------------
# 1. Data Loading Functions
# ---------------------------------------------------------------------

# Load RANS input features from file for each case
def loadCombinedArray(cases, field):
    data = np.concatenate([
        np.load(r'C:/Users/Mateo Gutierrez/OneDrive/Spring 2025/Machine Learning Engineering/Project/archive'
                + '/' + dataset + '/' + dataset + '_' + case + '_' + field + '.npy')
        for case in cases
    ])
    return data

# Load DNS labels (Reynolds stress) from file for each case
def loadTruthArray(cases, field):
    data = np.concatenate([
        np.load(r'C:/Users/Mateo Gutierrez/OneDrive/Spring 2025/Machine Learning Engineering/Project/archive/labels'
                + '/' + case + '_' + field + '.npy')
        for case in cases
    ])
    return data

# ---------------------------------------------------------------------
# 2. Load RANS and DNS/DATA
# ---------------------------------------------------------------------
dataset = 'komegasst'
case1 = ['CNDV_12600']

gradU = loadCombinedArray(case1, 'gradU')  
k = loadCombinedArray(case1, 'k')          
tau = loadTruthArray(case1, 'tau')         

# ---------------------------------------------------------------------
# 3. Compute Strain and Rotation Tensors
# ---------------------------------------------------------------------
gradU_T = np.transpose(gradU, (0, 2, 1))  
strain_rate = 0.5 * (gradU + gradU_T)    
rotation_rate = 0.5 * (gradU - gradU_T)  
N = strain_rate.shape[0]                  # Number of samples

# ---------------------------------------------------------------------
# 4. Compute Tensor Basis from S and Ω (Pope 1975)
# ---------------------------------------------------------------------
def compute_tensor_basis(S, Omega):
    N = S.shape[0]
    I = np.eye(3)[None, :, :]  
    T = np.zeros((N, 10, 3, 3))

    # Pope’s tensor basis
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
tensor_basis_torch = torch.tensor(T_basis, dtype=torch.float32)

# ---------------------------------------------------------------------
# 5. Compute Invariant Features (λ1 - λ5)
# ---------------------------------------------------------------------
I1 = np.trace(np.matmul(strain_rate, strain_rate), axis1=1, axis2=2)
I2 = np.trace(np.matmul(rotation_rate, rotation_rate), axis1=1, axis2=2)
I3 = np.trace(np.matmul(np.matmul(strain_rate, strain_rate), strain_rate), axis1=1, axis2=2)
I4 = np.trace(np.matmul(np.matmul(rotation_rate, rotation_rate), strain_rate), axis1=1, axis2=2)
I5 = np.trace(np.matmul(np.matmul(rotation_rate, strain_rate), rotation_rate), axis1=1, axis2=2)

# Stack features and normalize
input_features = np.stack([I1, I2, I3, I4, I5], axis=1)
features_mean = np.mean(input_features, axis=0)
features_std = np.std(input_features, axis=0) + 1e-20
features_tensor = torch.tensor((input_features - features_mean) / features_std, dtype=torch.float32)

# Ground-truth labels
actual_R_tensor = torch.tensor(tau, dtype=torch.float32)
k_tensor = torch.tensor(k, dtype=torch.float32)

# ---------------------------------------------------------------------
# 6. Define Dataset Class
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
        return (self.features[idx], self.tensor_basis[idx], self.R_target[idx], self.k_values[idx])

dataset_full = TBNNDataset(features_tensor, tensor_basis_torch, actual_R_tensor, k_tensor)

# ---------------------------------------------------------------------
# 7. Define the Tensor-Basis Neural Network
# ---------------------------------------------------------------------
class TBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_coeffs, output_dim, dropout_prob):
        super(TBNN, self).__init__()
        # Deep fully connected network
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),

        )
        self.coeff_layer = nn.Linear(hidden_dim, num_coeffs)  # 10 coefficients
        self.output_dim = output_dim

    def forward(self, x, tensor_basis):
        out = self.hidden(x)
        coeffs = self.coeff_layer(out).unsqueeze(-1).unsqueeze(-1)  # (batch, 10, 1, 1)
        return torch.sum(coeffs * tensor_basis, dim=1)  # Final anisotropic tensor a_ij

# ---------------------------------------------------------------------
# 8. Optuna Objective Function for Hyperparameter Tuning
# ---------------------------------------------------------------------
input_dim = 5
output_dim = 3

def objective(trial):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Initialize model and optimizer
    model = TBNN(input_dim, hidden_dim, 10, output_dim, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Split into train/val
    train_size = int(0.8 * N)
    val_size = N - train_size
    train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Train for fixed number of epochs
    for epoch in range(10):
        model.train()
        for batch_feats, batch_basis, batch_tau, batch_k in train_loader:
            optimizer.zero_grad()
            a_pred = model(batch_feats, batch_basis)
            k_exp = batch_k.unsqueeze(-1).unsqueeze(-1)
            I = torch.eye(3).unsqueeze(0).expand(a_pred.size(0), -1, -1)
            tau_pred = 2 * k_exp * (I / 3 + a_pred)  # Reconstruct full tau_ij
            loss = criterion(tau_pred, batch_tau)
            loss.backward()
            optimizer.step()

    # Validation MSE
    model.eval()
    mse = 0.0
    with torch.no_grad():
        for batch_feats, batch_basis, batch_tau, batch_k in val_loader:
            a_pred_val = model(batch_feats, batch_basis)
            k_val = batch_k.unsqueeze(-1).unsqueeze(-1)
            I = torch.eye(3).unsqueeze(0).expand(a_pred_val.size(0), -1, -1)
            tau_pred_val = 2 * k_val * (I / 3 + a_pred_val)
            mse += criterion(tau_pred_val, batch_tau).item() * batch_feats.size(0)

    return mse / len(val_loader.dataset)

# ---------------------------------------------------------------------
# 9. Run Optuna Hyperparameter Search
# ---------------------------------------------------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)

# Output best trial results
print("Best trial:")
print(f"  Value (MSE): {study.best_trial.value}")
for key, val in study.best_trial.params.items():
    print(f"  {key}: {val}")
