"""
Gaussian Process Surrogate Model
"""

import numpy as np
from typing import Tuple
from .base import SurrogateModel


class GaussianProcessSurrogate(SurrogateModel):
    """Gaussian Process regressor."""
    
    def __init__(self, kernel='rbf', alpha=1e-6):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
        except ImportError:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        if kernel == 'rbf':
            k = C(1.0) * RBF(1.0)
        else:
            k = RBF(1.0)
        
        self.model = GaussianProcessRegressor(
            kernel=k, 
            alpha=alpha, 
            normalize_y=True,
            n_restarts_optimizer=5
        )
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the Gaussian process model."""
        self.y_mean = float(y.mean())
        self.y_std = float(y.std() + 1e-8)
        y_n = (y - self.y_mean) / self.y_std
        self.model.fit(X, y_n)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and uncertainty."""
        mu, std = self.model.predict(X, return_std=True)
        mu = mu * self.y_std + self.y_mean
        std = std * self.y_std + 1e-8
        
        return mu, std