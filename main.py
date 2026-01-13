import argparse
from config import Config
from seed import set_seed
from BO import BONAS
from nasbench201_space import NASBench201Space
from dynamic_env import DynamicNASBenchmark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-path', type=str, required=True)
    parser.add_argument('--dynamic', action='store_true', help="Run Regime-Shift Experiment")
    parser.add_argument('--surrogate', type=str, default='mlp')
    parser.add_argument('--max-evals', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results.csv')
    args = parser.parse_args()
    
    cfg = Config()
    cfg.seed = args.seed
    cfg.max_evals = args.max_evals
    cfg.device = "cuda" # Force CUDA if available
    set_seed(cfg.seed)
    
    if args.dynamic:
        print("Initializing Dynamic Environment...")
        ss = DynamicNASBenchmark(args.api_path, switch_every=50)
    else:
        ss = NASBench201Space(args.api_path, dataset='cifar10')
        
    algo = BONAS(ss, cfg, surrogate_name=args.surrogate)
    results = algo.run()
    
    # Save results
    import csv
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "best_y", "idx"])
        for it, best, idx in results["history"]:
            w.writerow([it, best, idx])
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()