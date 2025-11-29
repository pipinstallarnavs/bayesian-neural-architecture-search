"""
Base class for all surrogate models.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class SurrogateModel(ABC):
    """Abstract base class for all surrogate models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fit the surrogate model to training data.
        
        Args:
            X: Training features [n_samples, n_features]
            y: Training targets [n_samples]
            **kwargs: Additional parameters (epochs, batch_size, etc.)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and uncertainty for input X.
        
        Args:
            X: Test features [n_samples, n_features]
            
        Returns:
            mu: Predicted means [n_samples]
            std: Predicted uncertainties [n_samples]
        """
        pass