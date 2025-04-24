
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import randint, uniform

# ---------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------
# Load RANS data from specified cases and fields
def loadCombinedArray(cases, field):
    return np.concatenate([
        np.load(f"C:/Users/Mateo Gutierrez/OneDrive/Spring 2025/Machine Learning Engineering/Project/archive/{dataset}/{dataset}_{case}_{field}.npy")
        for case in cases
    ])

# Load DNS/LES ground truth data
def loadTruthArray(cases, field):
    return np.concatenate([
        np.load(f"C:/Users/Mateo Gutierrez/OneDrive/Spring 2025/Machine Learning Engineering/Project/archive/labels/{case}_{field}.npy")
        for case in cases
    ])

# ---------------------------------------------------------------------
# Select Dataset and Load Features
# ---------------------------------------------------------------------
dataset = 'komegasst'
case1 = ['CNDV_12600']

gradU = loadCombinedArray(case1, 'gradU')     
k     = loadCombinedArray(case1, 'k')        
tau   = loadTruthArray(case1, 'tau')          

# ---------------------------------------------------------------------
# Compute Strain Rate and Rotation Rate Tensors
# ---------------------------------------------------------------------
gradU_T = np.transpose(gradU, (0, 2, 1))
strain_rate = 0.5 * (gradU + gradU_T)          
rotation_rate = 0.5 * (gradU - gradU_T)       
N = strain_rate.shape[0]                       # Number of samples

# ---------------------------------------------------------------------
# Compute Pope Tensor Basis (10 tensors)
# ---------------------------------------------------------------------
def compute_tensor_basis(S, Omega):
    N = S.shape[0]
    I = np.eye(3)[None, :, :]  
    T = np.zeros((N, 10, 3, 3))

    # Pope basis tensor definitions
    T[:, 0] = S
    T[:, 1] = np.matmul(S, Omega) - np.matmul(Omega, S)
    S2 = np.matmul(S, S)
    T[:, 2] = S2 - (1/3.) * I * np.trace(S2, axis1=1, axis2=2)[:, None, None]
    Omega2 = np.matmul(Omega, Omega)
    T[:, 3] = Omega2 - (1/3.) * I * np.trace(Omega2, axis1=1, axis2=2)[:, None, None]
    T[:, 4] = np.matmul(Omega, S2) - np.matmul(S2, Omega)
    SOmega2 = np.matmul(S, Omega2)
    T[:, 5] = np.matmul(Omega2, S) + np.matmul(S, Omega2) - (2/3.) * I * np.trace(SOmega2, axis1=1, axis2=2)[:, None, None]
    T[:, 6] = np.matmul(np.matmul(Omega, S), Omega2) - np.matmul(np.matmul(Omega2, S), Omega)
    T[:, 7] = np.matmul(np.matmul(S, Omega), S2) - np.matmul(np.matmul(S2, Omega), S)
    T[:, 8] = np.matmul(np.matmul(Omega, S2), Omega2) - np.matmul(np.matmul(Omega2, S2), Omega)
    T[:, 9] = np.matmul(np.matmul(S, Omega2), S2) - np.matmul(np.matmul(S2, Omega2), S)

    return T

T_basis = compute_tensor_basis(strain_rate, rotation_rate)

# ---------------------------------------------------------------------
# Compute Scalar Invariants (Lambdas 1–5)
# ---------------------------------------------------------------------
I1 = np.trace(np.matmul(strain_rate, strain_rate), axis1=1, axis2=2)
I2 = np.trace(np.matmul(rotation_rate, rotation_rate), axis1=1, axis2=2)
I3 = np.trace(np.matmul(np.matmul(strain_rate, strain_rate), strain_rate), axis1=1, axis2=2)
I4 = np.trace(np.matmul(np.matmul(rotation_rate, rotation_rate), strain_rate), axis1=1, axis2=2)
I5 = np.trace(np.matmul(np.matmul(rotation_rate, strain_rate), rotation_rate), axis1=1, axis2=2)

# Stack scalar invariants as input features
input_features = np.stack([I1, I2, I3, I4, I5], axis=1)

# ---------------------------------------------------------------------
# Normalize Features and Prepare Input Tensors
# ---------------------------------------------------------------------
features_mean = np.mean(input_features, axis=0)
features_std = np.std(input_features, axis=0) + 1e-20

# Normalize scalar inputs
features_tensor = torch.tensor((input_features - features_mean) / features_std, dtype=torch.float32)

# Flatten tensor basis and concatenate with invariants
tensor_basis_flat = torch.tensor(T_basis.reshape(N, -1), dtype=torch.float32)
X_full = torch.cat([features_tensor, tensor_basis_flat], dim=1).numpy()

# Flatten tau tensor as target
y_full = torch.tensor(tau, dtype=torch.float32).view(N, -1).numpy()

# ---------------------------------------------------------------------
# PyTorch TBNN Architecture
# ---------------------------------------------------------------------
class TBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob):
        super(TBNN, self).__init__()
        # Deep feedforward network with dropout and LeakyReLU activations
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout_prob),

        )
        self.output_layer = nn.Linear(hidden_dim, 9)  # Output flattened 3x3 tau tensor

    def forward(self, x):
        return self.output_layer(self.hidden(x))

# ---------------------------------------------------------------------
# Wrapper for Scikit-learn Compatibility with RandomizedSearchCV
# ---------------------------------------------------------------------
class TBNNWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_dim=128, dropout_prob=0.2, lr=1e-4, batch_size=64, epochs=10):
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def fit(self, X, y):
        # Wrap data as PyTorch dataset
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Instantiate model and optimizer
        self.model = TBNN(X.shape[1], self.hidden_dim, self.dropout_prob)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Training loop
        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        # Inference mode: no gradients needed
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            pred = self.model(X_tensor)
        return pred.detach().numpy()

# ---------------------------------------------------------------------
# Hyperparameter Search Setup
# ---------------------------------------------------------------------
# Define scoring function for RandomizedSearchCV
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Define random search parameter distributions
param_dist = {
    'hidden_dim': randint(64, 256),
    'dropout_prob': uniform(0.1, 0.5),
    'lr': uniform(1e-5, 1e-3),
    'batch_size': [32, 64, 128, 256],
    'epochs': [10]
}

# Perform random search using cross-validation
model = TBNNWrapper()
search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5,
                            scoring=scorer, cv=3, verbose=2, random_state=42)

search.fit(X_full, y_full)

# Display best hyperparameters and MSE
print("Best Params:", search.best_params_)
print("Best MSE: ", -search.best_score_)
