"""
Gradient Boosting Machine Surrogate Model
"""

import numpy as np
from typing import Tuple
from .base import SurrogateModel


class GradientBoostingSurrogate(SurrogateModel):
    """Gradient Boosting regressor using scikit-learn."""
    
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1):
        try:
            from sklearn.ensemble import GradientBoostingRegressor
        except ImportError:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbose=0
        )
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the gradient boosting model."""
        self.y_mean = float(y.mean())
        self.y_std = float(y.std() + 1e-8)
        y_n = (y - self.y_mean) / self.y_std
        self.model.fit(X, y_n)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and uncertainty."""
        mu = self.model.predict(X) * self.y_std + self.y_mean
        
        # Estimate uncertainty from individual tree predictions
        predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_.flatten()
        ])
        std = predictions.std(axis=0) * self.y_std + 1e-8
        
        return mu, std