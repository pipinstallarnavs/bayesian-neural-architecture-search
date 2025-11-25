# acquisition.py
import numpy as np
from scipy.stats import norm

def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float = 0.0) -> np.ndarray:
    """EI for maximization."""
    sigma = np.maximum(sigma, 1e-12)
    imp = mu - best_y - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei

def ucb(mu: np.ndarray, sigma: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """UCB for maximization."""
    return mu + np.sqrt(beta) * sigma
