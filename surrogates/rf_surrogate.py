"""
Random Forest Surrogate Model
"""

import numpy as np
from typing import Tuple
from .base import SurrogateModel


class RandomForestSurrogate(SurrogateModel):
    """Random Forest regressor."""
    
    def __init__(self, n_estimators=100, max_depth=10):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the random forest model."""
        self.y_mean = float(y.mean())
        self.y_std = float(y.std() + 1e-8)
        y_n = (y - self.y_mean) / self.y_std
        self.model.fit(X, y_n)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and uncertainty."""
        mu = self.model.predict(X) * self.y_std + self.y_mean
        
        # Estimate std from individual tree predictions
        predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ])
        std = predictions.std(axis=0) * self.y_std + 1e-8
        
        return mu, std