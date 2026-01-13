import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from .base import SurrogateModel

class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.relu(self.bn2(self.conv2(self.dropout(self.relu(self.bn1(self.conv1(x)))))))
        out += self.residual(x)
        return self.relu(out)

class CNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=64):
        super().__init__()
        self.start_conv = nn.Conv1d(1, hidden_dim, 3, padding=1)
        self.layer1 = ResNet1DBlock(hidden_dim, hidden_dim)
        self.layer2 = ResNet1DBlock(hidden_dim, hidden_dim*2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        # x starts as [Batch, Input_Dim]
        x = x.unsqueeze(1) # [Batch, 1, Input_Dim] (Channels=1, Length=Input_Dim)
        x = self.start_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

class CNNEnsemble(SurrogateModel):
    def __init__(self, input_dim: int, hidden_dim=64, ensemble_size=5, device="cuda", lr=1e-3, **kwargs):
        self.device = device
        self.members = []
        self.optims = []
        for _ in range(ensemble_size):
            model = CNNRegressor(input_dim, hidden_dim).to(device)
            self.members.append(model)
            self.optims.append(optim.Adam(model.parameters(), lr=lr))
        self.loss_fn = nn.MSELoss()
        self.y_mean, self.y_std = 0.0, 1.0

    def fit(self, X, y, epochs=100, batch_size=64, seed=0, **kwargs):
        torch.manual_seed(seed)
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_raw = y.astype(np.float32)
        self.y_mean, self.y_std = float(y_raw.mean()), float(y_raw.std() + 1e-8)
        y_n = (y_raw - self.y_mean) / self.y_std
        y_t = torch.tensor(y_n, dtype=torch.float32, device=self.device)
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        
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

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        preds = []
        with torch.no_grad():
            for m in self.members:
                m.eval()
                preds.append(m(X_t).cpu().numpy())
        preds = np.stack(preds, axis=0) # [Ensemble, N]
        
        # De-normalize
        mu = preds.mean(axis=0) * self.y_std + self.y_mean
        std = preds.std(axis=0) * self.y_std + 1e-8
        return mu, std