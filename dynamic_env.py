from nasbench201_space import NASBench201Space

class DynamicNASBenchmark:
    """
    Simulates non-stationary environment.
    Regime 0: CIFAR-10
    Regime 1: CIFAR-100
    Regime 2: ImageNet16-120
    """
    def __init__(self, api_path, switch_every=50):
        self.switch_every = switch_every
        self.datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
        # Load all datasets
        self.spaces = {
            d: NASBench201Space(api_path, dataset=d) for d in self.datasets
        }
        self.current_step = 0
        self.current_regime = 0
        # Pool is identical across datasets in NAS-Bench-201
        self.pool = self.spaces['cifar10'].enumerate()
        # Proxy for encoding methods
        self.OPS = self.spaces['cifar10'].OPS
        
    def enumerate(self):
        return self.pool

    def encode(self, arch):
        return self.spaces['cifar10'].encode(arch)
        
    def encode_graph(self, arch):
        return self.spaces['cifar10'].encode_graph(arch)

    def evaluate(self, arch):
        # Determine regime
        regime_idx = (self.current_step // self.switch_every) % len(self.datasets)
        dataset = self.datasets[regime_idx]
        
        if regime_idx != self.current_regime:
            print(f"\n[!!!] MARKET SHIFT: Switching to {dataset} [!!!]\n")
            self.current_regime = regime_idx
            
        y = self.spaces[dataset].evaluate(arch)
        self.current_step += 1
        return y