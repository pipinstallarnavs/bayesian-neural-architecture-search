import numpy as np
from surrogates import get_surrogate_model
from tests import expected_improvement, ucb

class BONAS:
    def __init__(self, search_space, config, surrogate_name='mlp'):
        self.ss = search_space
        self.cfg = config
        self.surrogate_name = surrogate_name
        self.hist = [] 
        
        print("Pre-encoding search space (Vectorized)...")
        self.pool = self.ss.enumerate()
        
        # 1. Pre-compute Flat Encodings (for MLP, CNN, etc)
        self.X_pool_flat = np.stack([self.ss.encode(a) for a in self.pool], axis=0).astype(np.float32)
        
        # 2. Pre-compute Graph Encodings (for GNN)
        graph_data = [self.ss.encode_graph(a) for a in self.pool]
        self.X_pool_ops = np.stack([g[0] for g in graph_data], axis=0)
        self.X_pool_adjs = np.stack([g[1] for g in graph_data], axis=0)
        
        self.seen_mask = np.zeros(len(self.pool), dtype=bool)
        self.train_idxs = []
        self.y_train = []

        # Initialize Surrogate
        self.model = get_surrogate_model(
            surrogate_name,
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
        
        while self.seen_mask.sum() < total_allowed:
            # 1. Fit
            X_train = self._get_data(self.train_idxs)
            y_train = np.array(self.y_train)
            self.model.fit(X_train, y_train, epochs=self.cfg.epochs, verbose=False)
            
            # 2. Predict Unseen
            unseen_idxs = np.where(~self.seen_mask)[0]
            if len(unseen_idxs) == 0: break
            
            X_cand = self._get_data(unseen_idxs)
            mu, std = self.model.predict(X_cand)
            
            # 3. Acquisition
            best_y = np.max(y_train)
            if self.cfg.acquisition == 'ei':
                score = expected_improvement(mu, std, best_y)
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
            
            self.hist.append((self.seen_mask.sum(), max(self.y_train), global_idx))
            print(f"Iter {self.seen_mask.sum()} | Best: {max(self.y_train):.4f}% | Model: {self.surrogate_name}")

        return {"best_y": max(self.y_train), "history": self.hist}