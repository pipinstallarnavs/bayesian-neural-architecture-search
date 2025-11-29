"""
MLP Ensemble Surrogate Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from .base import SurrogateModel


@dataclass
class MLPConfig:
    input_dim: int
    hidden_sizes: Tuple[int, ...] = (128, 128)
    dropout: float = 0.1


class MLPRegressor(nn.Module):
    """Single MLP regressor."""
    
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPEnsemble(SurrogateModel):
    """Deep ensemble of MLP regressors."""
    
    def __init__(self, input_dim: int, hidden_sizes=(128, 128), dropout=0.1, 
                 ensemble_size=5, device="cpu", lr=1e-3):
        self.device = device
        self.members = []
        self.optims = []
        for _ in range(ensemble_size):
            model = MLPRegressor(MLPConfig(input_dim=input_dim, hidden_sizes=hidden_sizes, 
                                          dropout=dropout)).to(device)
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
                    print(f"[MLP Ensemble] Early stop at epoch {ep}")
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