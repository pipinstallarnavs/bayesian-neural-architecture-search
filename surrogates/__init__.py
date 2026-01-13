from .base import SurrogateModel
from .mlp_surrogate import MLPEnsemble
from .cnn_surrogate import CNNEnsemble
from .resnet_surrogate import ResNetEnsemble
from .gbm_surrogate import GradientBoostingSurrogate
from .rf_surrogate import RandomForestSurrogate
from .gp_surrogate import GaussianProcessSurrogate
from .attention_surrogate import AttentionEnsemble
from .gnn_surrogate import GNNEnsemble  # <--- NEW

SURROGATE_MODELS = {
    'mlp': MLPEnsemble,
    'cnn': CNNEnsemble,
    'resnet': ResNetEnsemble,
    'gbm': GradientBoostingSurrogate,
    'rf': RandomForestSurrogate,
    'gp': GaussianProcessSurrogate,
    'attention': AttentionEnsemble,
    'gnn': GNNEnsemble,  # <--- NEW
}

def get_surrogate_model(name: str, **kwargs):
    if name not in SURROGATE_MODELS:
        raise ValueError(f"Unknown: {name}")
    return SURROGATE_MODELS[name](**kwargs)