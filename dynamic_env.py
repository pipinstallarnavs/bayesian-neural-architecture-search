from nasbench201_space import NASBench201Space

class DynamicNASBenchmark:
    """
    Optimized: Loads the API only once and switches the query dataset key.
    """
    def __init__(self, api_path, switch_every=50):
        self.switch_every = switch_every
        self.datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
        
        print(f"Loading NATS-Bench API from {api_path}...")
        # Initialize ONE space object. This triggers the heavy load once.
        self.master_space = NASBench201Space(api_path, dataset='cifar10')
        
        self.current_step = 0
        self.current_regime = 0
        self.pool = self.master_space.enumerate()
        self.OPS = self.master_space.OPS
        
    def enumerate(self):
        return self.pool

    def encode(self, arch):
        return self.master_space.encode(arch)
        
    def encode_graph(self, arch):
        return self.master_space.encode_graph(arch)

    def evaluate(self, arch):
        # Determine regime
        regime_idx = (self.current_step // self.switch_every) % len(self.datasets)
        dataset = self.datasets[regime_idx]
        
        if regime_idx != self.current_regime:
            print(f"\n[!!!] MARKET SHIFT: Switching to {dataset} [!!!]\n")
            self.current_regime = regime_idx
            
        # HACK: Manually swap the dataset string in the master space
        self.master_space.dataset = dataset
        
        # Now evaluate using that dataset
        y = self.master_space.evaluate(arch)
        self.current_step += 1
        return y