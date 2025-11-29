"""
Surrogate Models Package for Bayesian Optimization.
Provides various models for NAS surrogate modeling.
"""

from .base import SurrogateModel
from .mlp_surrogate import MLPEnsemble
from .cnn_surrogate import CNNEnsemble
from .resnet_surrogate import ResNetEnsemble
from .gbm_surrogate import GradientBoostingSurrogate
from .rf_surrogate import RandomForestSurrogate
from .gp_surrogate import GaussianProcessSurrogate
from .attention_surrogate import AttentionEnsemble

__all__ = [
    'SurrogateModel',
    'MLPEnsemble',
    'CNNEnsemble',
    'ResNetEnsemble',
    'GradientBoostingSurrogate',
    'RandomForestSurrogate',
    'GaussianProcessSurrogate',
    'AttentionEnsemble',
    'get_surrogate_model',
]

SURROGATE_MODELS = {
    'mlp': MLPEnsemble,
    'cnn': CNNEnsemble,
    'resnet': ResNetEnsemble,
    'gbm': GradientBoostingSurrogate,
    'rf': RandomForestSurrogate,
    'gp': GaussianProcessSurrogate,
    'attention': AttentionEnsemble,
}


def get_surrogate_model(name: str, **kwargs):
    """
    Factory function to get surrogate model by name.
    
    Args:
        name: Model name ('mlp', 'cnn', 'resnet', 'gbm', 'rf', 'gp', 'attention')
        **kwargs: Arguments to pass to the model constructor
        
    Returns:
        Instantiated surrogate model
        
    Raises:
        ValueError: If model name is not recognized
        
    Example:
        >>> model = get_surrogate_model('cnn', input_dim=30, device='cuda')
        >>> model.fit(X_train, y_train)
        >>> mu, std = model.predict(X_test)
    """
    if name not in SURROGATE_MODELS:
        raise ValueError(
            f"Unknown surrogate model: {name}. "
            f"Available models: {list(SURROGATE_MODELS.keys())}"
        )
    return SURROGATE_MODELS[name](**kwargs)