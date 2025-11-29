import argparse
from config import Config
from seed import set_seed
from BO import BONAS

def main():
    parser = argparse.ArgumentParser(description='BO-NAS on NAS-Bench-201')
    parser.add_argument('--api-path', type=str, required=True,
                       help='Path to NATS-Bench benchmark')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'ImageNet16-120'],
                       help='Dataset to use')
    parser.add_argument('--max-evals', type=int, default=100,
                       help='Total architecture evaluations')
    parser.add_argument('--init-random', type=int, default=20,
                       help='Random initialization samples')
    parser.add_argument('--fast', action='store_true',
                       help='Use 12-epoch results (faster)')
    parser.add_argument('--acquisition', type=str, default='ei',
                       choices=['ei', 'ucb'], help='Acquisition function')
    parser.add_argument('--surrogate', type=str, default='mlp',
                       choices=['mlp', 'cnn', 'resnet', 'gbm', 'rf', 'gp', 'attention'],
                       help='Surrogate model to use')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results.csv')
    
    args = parser.parse_args()
    
    # Setup
    cfg = Config()
    cfg.seed = args.seed
    cfg.max_evals = args.max_evals
    cfg.init_random = args.init_random
    cfg.acquisition = args.acquisition
    set_seed(cfg.seed)
    
    # Create search space
    print(f"\n{'='*60}")
    print(f"BO-NAS on NAS-Bench-201")
    print(f"{'='*60}")
    
    from nasbench201_space import NASBench201Space
    ss = NASBench201Space(
        api_path=args.api_path,
        dataset=args.dataset,
        use_12epoch=args.fast
    )
    cfg.input_dim = ss.input_dim
    
    print(f"Dataset: {args.dataset}")
    print(f"Search space size: {len(ss.enumerate())}")
    print(f"Input dimension: {cfg.input_dim}")
    print(f"Max evaluations: {cfg.max_evals}")
    print(f"Acquisition: {cfg.acquisition}")
    print(f"Surrogate model: {args.surrogate.upper()}")
    
    # Run BO
    print(f"\n{'='*60}")
    print("Running Bayesian Optimization...")
    print(f"{'='*60}\n")
    
    algo = BONAS(ss, cfg, surrogate_name=args.surrogate)
    results = algo.run()
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Surrogate model: {results['surrogate_model'].upper()}")
    print(f"Total evaluations: {results['n_evals']}")
    print(f"Best accuracy    : {results['best_y']:.4f}%")
    print(f"Best arch index  : {results['best_idx']}")
    
    # Save
    import csv
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "best_so_far", "arch_idx"])
        for it, best, idx in results["history"]:
            w.writerow([it, best, idx])
    print(f"\nSaved: {args.output}")

if __name__ == "__main__":
    main()