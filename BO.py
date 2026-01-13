import numpy as np
from scipy.stats import kendalltau
from surrogates import get_surrogate_model
from tests import expected_improvement, ucb

class BONAS:
    def __init__(self, search_space, config, surrogate_name='mlp', reset_on_shift=False):
        self.ss = search_space
        self.cfg = config
        self.surrogate_name = surrogate_name
        self.reset_on_shift = reset_on_shift
        self.hist = [] 
        
        print("Pre-encoding search space (Vectorized)...")
        self.pool = self.ss.enumerate()
        
        # 1. Pre-compute Flat Encodings
        self.X_pool_flat = np.stack([self.ss.encode(a) for a in self.pool], axis=0).astype(np.float32)
        
        # 2. Pre-compute Graph Encodings
        graph_data = [self.ss.encode_graph(a) for a in self.pool]
        self.X_pool_ops = np.stack([g[0] for g in graph_data], axis=0)
        self.X_pool_adjs = np.stack([g[1] for g in graph_data], axis=0)
        
        self.seen_mask = np.zeros(len(self.pool), dtype=bool)
        self.train_idxs = []
        self.y_train = []

        # Initialize Surrogate
        self._init_model()

    def _init_model(self):
        """Helper to initialize (or re-initialize) the model."""
        print(f"[{self.surrogate_name.upper()}] Initializing new model weights...")
        self.model = get_surrogate_model(
            self.surrogate_name,
            input_dim=self.X_pool_flat.shape[1],
            num_ops=len(self.ss.OPS),
            hidden_dim=128,
            ensemble_size=self.cfg.ensemble_size,
            device=self.cfg.device,
            lr=self.cfg.lr
        )

    def _get_data(self, idxs):
        """Helper to get correct data format for surrogate."""
        if self.surrogate_name == 'gnn':
            return {
                "ops": self.X_pool_ops[idxs],
                "adjs": self.X_pool_adjs[idxs]
            }
        return self.X_pool_flat[idxs]

    def warm_start(self):
        idxs = np.random.choice(len(self.pool), size=min(self.cfg.init_random, len(self.pool)), replace=False)
        for i in idxs:
            y = self.ss.evaluate(self.pool[i])
            self.seen_mask[i] = True
            self.train_idxs.append(i)
            self.y_train.append(y)

    def run(self):
        self.warm_start()
        total_allowed = min(self.cfg.max_evals, len(self.pool))
        
        # Determine switch interval from dynamic env (if applicable)
        switch_interval = getattr(self.ss, 'switch_every', None)

        while self.seen_mask.sum() < total_allowed:
            curr_iter = self.seen_mask.sum()

            # --- EXPERIMENTAL CONTROL: RESET ON SHIFT ---
            if self.reset_on_shift and switch_interval and (curr_iter % switch_interval == 0):
                print(f"\n[CONTROL] Resetting Surrogate Model at iter {curr_iter} (Market Shift detected)\n")
                self._init_model()

            # 1. Fit
            X_train = self._get_data(self.train_idxs)
            y_train = np.array(self.y_train)
            self.model.fit(X_train, y_train, epochs=self.cfg.epochs, verbose=False)
            
            # 2. Predict Unseen (for acquisition)
            unseen_idxs = np.where(~self.seen_mask)[0]
            if len(unseen_idxs) == 0: break
            
            X_cand = self._get_data(unseen_idxs)
            mu, std = self.model.predict(X_cand)
            
            # --- PAPER METRIC: KENDALL'S TAU ---
            # Measure how well we rank the unseen candidates compared to ground truth (if available)
            # NOTE: In a real setting, we don't have 'true_y_unseen'. 
            # But for research benchmarking, we cheat slightly to calculate the metric.
            # Only calculate on a subset to save time if pool is huge.
            if len(unseen_idxs) > 0:
                # evaluate() is cheap in NAS-Bench-201 (lookup table)
                # We sample 200 random points to estimate Tau to avoid slowing down too much
                eval_subset = np.random.choice(unseen_idxs, size=min(200, len(unseen_idxs)), replace=False)
                # Note: This evaluate() call is strictly for METRICS, not for TRAINING data.
                true_y_subset = [self.ss.evaluate(self.pool[i]) for i in eval_subset]
                
                # Get predictions for these specific ones
                # We need to re-predict or map indices. Let's just predict this subset:
                X_subset = self._get_data(eval_subset)
                mu_subset, _ = self.model.predict(X_subset)
                
                tau, _ = kendalltau(mu_subset, true_y_subset)
            else:
                tau = 0.0

            # 3. Acquisition
            best_y_train = np.max(y_train)
            if self.cfg.acquisition == 'ei':
                score = expected_improvement(mu, std, best_y_train)
            else:
                score = ucb(mu, std, beta=self.cfg.beta)
                
            # 4. Select
            best_local = np.argmax(score)
            global_idx = unseen_idxs[best_local]
            
            # 5. Evaluate
            y_new = self.ss.evaluate(self.pool[global_idx])
            self.seen_mask[global_idx] = True
            self.train_idxs.append(global_idx)
            self.y_train.append(y_new)
            
            # Metric: Regret (Best Possible - Best Found So Far)
            # In dynamic envs, "Best Possible" changes, so we assume 100% or approximate
            # For NAS-Bench-201, we know the global best is around 94-95% usually.
            # We will just log "Best Found" and let the plotter handle regret.
            
            self.hist.append({
                "iter": int(self.seen_mask.sum()),
                "best_y": float(max(self.y_train)),
                "idx": int(global_idx),
                "tau": float(tau),
                "last_y": float(y_new)
            })
            
            print(f"Iter {self.seen_mask.sum()} | Best: {max(self.y_train):.4f}% | Tau: {tau:.4f} | Model: {self.surrogate_name}")

        return {"best_y": max(self.y_train), "history": self.hist}