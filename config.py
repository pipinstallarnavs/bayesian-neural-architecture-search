import torch
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_evals: int = 100         # total architecture evaluations
    init_random: int = 20        # random warmup evaluations
    batch_size: int = 64
    lr: float = 1e-3             # learning rate
    epochs: int = 200            # surrogate training epochs per BO iteration
    patience: int = 20           # early stopping patience (epochs)
    ensemble_size: int = 5       # MLP ensemble members
    hidden_sizes: tuple = (128, 128)
    dropout: float = 0.1
    acquisition: str = "ei"      # "ei" or "ucb"
    beta: float = 2.0            # UCB parameter
    y_noise: float = 1e-3        # small jitter when fitting targets
    normalize_y: bool = True

    # Candidate sampling per iteration (if we can't enumerate all)
    candidates_per_iter: int = 2000

    # === NAS-specific ===
    # If using NAS-Bench-201, keep None; the wrapper will set these.
    input_dim: int = None