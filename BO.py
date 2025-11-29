# bo_loop.py
import numpy as np
from typing import Dict, Any, List, Tuple, Callable
from surrogates import get_surrogate_model
from tests import expected_improvement, ucb


class BONAS:
    def __init__(self, search_space, config, surrogate_name='mlp'):
        self.ss = search_space
        self.cfg = config
        self.surrogate_name = surrogate_name
        self.X_train = []  # encoded vectors
        self.y_train = []
        self.hist = []     # (iter, best_y, arch_idx)
        self.pool = self.ss.enumerate()
        self.seen_mask = np.zeros(len(self.pool), dtype=bool)

        input_dim = self.ss.input_dim if hasattr(self.ss, "input_dim") else self.cfg.input_dim
        
        # Create surrogate model
        self.model = get_surrogate_model(
            surrogate_name,
            input_dim=input_dim,
            hidden_sizes=self.cfg.hidden_sizes,
            dropout=self.cfg.dropout,
            ensemble_size=self.cfg.ensemble_size,
            device=self.cfg.device,
            lr=self.cfg.lr,
        )

    def _encode_many(self, archs: List[Dict[str, Any]]) -> np.ndarray:
        return np.stack([self.ss.encode(a) for a in archs], axis=0)

    def _sample_candidates(self) -> Tuple[np.ndarray, np.ndarray]:
        idxs = np.where(~self.seen_mask)[0]
        if len(idxs) == 0:
            return np.array([], dtype=int), np.zeros((0, self.ss.input_dim), dtype=np.float32)
        if len(idxs) > self.cfg.candidates_per_iter:
            idxs = np.random.choice(idxs, size=self.cfg.candidates_per_iter, replace=False)
        archs = [self.pool[i] for i in idxs]
        X = self._encode_many(archs)
        return idxs, X

    def warm_start(self):
        idxs = np.random.choice(np.arange(len(self.pool)), size=min(self.cfg.init_random, len(self.pool)), replace=False)
        for i in idxs:
            y = self.ss.evaluate(self.pool[i])
            self.seen_mask[i] = True
            self.X_train.append(self.ss.encode(self.pool[i]))
            self.y_train.append(y)
        self.X_train = np.array(self.X_train, dtype=np.float32)
        self.y_train = np.array(self.y_train, dtype=np.float32)

    def _fit_surrogate(self, seed_offset=0):
        self.model.fit(
            self.X_train, self.y_train,
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            patience=self.cfg.patience,
            y_noise=self.cfg.y_noise,
            normalize_y=self.cfg.normalize_y,
            verbose=False,
            seed=self.cfg.seed + seed_offset
        )

    def _choose_next(self) -> int:
        cand_idxs, Xc = self._sample_candidates()
        if Xc.shape[0] == 0:
            return -1

        mu, std = self.model.predict(Xc)
        best_y = float(np.max(self.y_train))
        if self.cfg.acquisition.lower() == "ei":
            score = expected_improvement(mu, std, best_y)
        elif self.cfg.acquisition.lower() == "ucb":
            score = ucb(mu, std, beta=self.cfg.beta)
        else:
            raise ValueError("Unknown acquisition")

        j = int(np.argmax(score))
        return int(cand_idxs[j])

    def run(self):
        # Warmup
        self.warm_start()

        # BO iterations
        total_allowed = min(self.cfg.max_evals, len(self.pool))
        while self.seen_mask.sum() < total_allowed:
            self._fit_surrogate(seed_offset=self.seen_mask.sum())
            idx = self._choose_next()
            if idx < 0:
                break
            y = self.ss.evaluate(self.pool[idx])
            self.seen_mask[idx] = True
            self.X_train = np.vstack([self.X_train, self.ss.encode(self.pool[idx])])
            self.y_train = np.append(self.y_train, y)
            self.hist.append((int(self.seen_mask.sum()), float(self.y_train.max()), idx))

        return {
            "best_y": float(self.y_train.max()),
            "best_idx": int(np.argmax(self.y_train)),
            "history": self.hist,
            "n_evals": int(self.seen_mask.sum()),
            "surrogate_model": self.surrogate_name,
        }