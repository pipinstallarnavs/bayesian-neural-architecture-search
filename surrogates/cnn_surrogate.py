"""
CNN Ensemble Surrogate Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from .base import SurrogateModel


class CNNRegressor(nn.Module):
    """1D CNN for feature extraction."""
    
    def __init__(self, input_dim: int, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, input_dim] -> [batch, 1, input_dim]
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class CNNEnsemble(SurrogateModel):
    """Ensemble of 1D CNN regressors."""
    
    def __init__(self, input_dim: int, hidden_dim=64, ensemble_size=5, 
                 device="cpu", lr=1e-3):
        self.device = device
        self.members = []
        self.optims = []
        for _ in range(ensemble_size):
            model = CNNRegressor(input_dim, hidden_dim).to(device)
            self.members.append(model)
            self.optims.append(optim.Adam(model.parameters(), lr=lr))
        self.loss_fn = nn.MSELoss()
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=200, batch_size=64, 
            patience=20, y_noise=1e-3, normalize_y=True, verbose=False, seed=0):
        torch.manual_seed(seed)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_raw = y.astype(np.float32)
        
        if normalize_y:
            self.y_mean = float(y_raw.mean())
            self.y_std = float(y_raw.std() + 1e-8)
            y_n = (y_raw - self.y_mean) / self.y_std
        else:
            self.y_mean, self.y_std = 0.0, 1.0
            y_n = y_raw
        
        y_t = torch.tensor(y_n + np.random.normal(0, y_noise, size=y_n.shape).astype(np.float32), 
                          device=self.device)
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        best_losses = [float("inf")] * len(self.members)
        waits = [0] * len(self.members)

        for ep in range(epochs):
            for m, opt in zip(self.members, self.optims):
                m.train()
            for xb, yb in dl:
                for m, opt in zip(self.members, self.optims):
                    opt.zero_grad()
                    pred = m(xb)
                    loss = self.loss_fn(pred, yb)
                    loss.backward()
                    opt.step()

            with torch.no_grad():
                for i, m in enumerate(self.members):
                    m.eval()
                    pred = m(X_t)
                    loss = self.loss_fn(pred, y_t).item()
                    if loss + 1e-9 < best_losses[i]:
                        best_losses[i] = loss
                        waits[i] = 0
                    else:
                        waits[i] += 1

            if patience and max(waits) > patience:
                if verbose:
                    print(f"[CNN Ensemble] Early stop at epoch {ep}")
                break

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        preds = []
        with torch.no_grad():
            for m in self.members:
                m.eval()
                p = m(X_t).cpu().numpy()
                preds.append(p)
        preds = np.stack(preds, axis=0)
        preds = preds * self.y_std + self.y_mean
        mu = preds.mean(axis=0)
        std = preds.std(axis=0) + 1e-8
        return mu, std