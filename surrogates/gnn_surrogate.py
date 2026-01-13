import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from .base import SurrogateModel

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x: [batch, nodes, in]
        # adj: [batch, nodes, nodes]
        support = self.linear(x)
        output = torch.bmm(adj, support) # Graph Conv
        return output

class GNNRegressor(nn.Module):
    def __init__(self, num_ops, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_ops, hidden_dim)
        self.gc1 = GCNLayer(hidden_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, hidden_dim)
        self.gc3 = GCNLayer(hidden_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_ops, adj):
        h = self.embedding(x_ops) # [B, 6, H]
        h = F.relu(self.gc1(h, adj)) + h
        h = self.dropout(h)
        h = F.relu(self.gc2(h, adj)) + h
        h = self.dropout(h)
        h = F.relu(self.gc3(h, adj)) + h
        
        # Global Average Pooling
        h = torch.mean(h, dim=1) 
        return self.fc(h).squeeze(-1)

class GNNEnsemble(SurrogateModel):
    def __init__(self, num_ops=5, hidden_dim=128, ensemble_size=5, device="cuda", lr=1e-3, **kwargs):
        self.device = device
        self.members = []
        self.optims = []
        for _ in range(ensemble_size):
            model = GNNRegressor(num_ops, hidden_dim).to(device)
            self.members.append(model)
            self.optims.append(optim.Adam(model.parameters(), lr=lr))
        self.loss_fn = nn.MSELoss()
        self.y_mean, self.y_std = 0.0, 1.0

    def fit(self, X_data, y, epochs=100, batch_size=64, patience=15, **kwargs):
        # Unpack
        ops, adjs = X_data["ops"], X_data["adjs"]
        
        y_raw = y.astype(np.float32)
        self.y_mean, self.y_std = float(y_raw.mean()), float(y_raw.std() + 1e-8)
        y_n = (y_raw - self.y_mean) / self.y_std
        
        ops_t = torch.tensor(ops, dtype=torch.long, device=self.device)
        adjs_t = torch.tensor(adjs, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_n, dtype=torch.float32, device=self.device)
        
        ds = torch.utils.data.TensorDataset(ops_t, adjs_t, y_t)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

        for ep in range(epochs):
            for m, opt in zip(self.members, self.optims):
                m.train()
            for b_ops, b_adj, b_y in dl:
                for m, opt in zip(self.members, self.optims):
                    opt.zero_grad()
                    pred = m(b_ops, b_adj)
                    loss = self.loss_fn(pred, b_y)
                    loss.backward()
                    opt.step()

    def predict(self, X_data) -> Tuple[np.ndarray, np.ndarray]:
        ops, adjs = X_data["ops"], X_data["adjs"]
        ops_t = torch.tensor(ops, dtype=torch.long, device=self.device)
        adjs_t = torch.tensor(adjs, dtype=torch.float32, device=self.device)
        
        preds = []
        with torch.no_grad():
            for m in self.members:
                m.eval()
                p = m(ops_t, adjs_t).cpu().numpy()
                preds.append(p)
        
        preds = np.stack(preds, axis=0)
        preds = preds * self.y_std + self.y_mean
        return preds.mean(axis=0), preds.std(axis=0) + 1e-8