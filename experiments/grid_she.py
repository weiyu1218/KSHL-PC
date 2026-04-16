import argparse
import itertools
import subprocess
import json
import csv
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search for SHE hyperparameters (Mann only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data and model parameters
    parser.add_argument('--dataset', type=str, default='Mann',
                        choices=['Mann'],
                        help="PPI dataset (fixed to Mann for grid search)")
    parser.add_argument('--model', type=str, default='SHE',
                        choices=['SHE'],
                        help="Model to use")

    # Search space parameters
    parser.add_argument('--r_list', type=int, nargs='+', default=[64, 96, 128],
                        help="List of r values (SVD rank)")
    parser.add_argument('--k_factor_list', type=float, nargs='+', default=[1.0, 1.5],
                        help="List of factors to compute she_k: she_k = ceil(k_factor * r); if k_factor=1.0 -> max(r, 64)")
    parser.add_argument('--T_list', type=int, nargs='+', default=[10],
                        help="List of SHE window sizes")
    parser.add_argument('--alpha_list', type=float, nargs='+', default=[0.05, 0.1],
                        help="List of SHE alpha values")

    # Run control
    parser.add_argument('--n_repeats', type=int, default=3,
                        help="Number of repeats with different negative sampling")
    parser.add_argument('--folds', type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--max_workers', type=int, default=1,
                        help="Maximum number of parallel workers")
    parser.add_argument('--resume', action='store_true',
                        help="Resume from existing results (skip completed configs)")

    # Output
    parser.add_argument('--out_dir', type=str, default='results/grid_she',
                        help="Output directory for logs and results")
    parser.add_argument('--summary_csv', type=str, default=None,
                        help="Path to summary CSV file (default: out_dir/summary_she.csv)")
    parser.add_argument('--best_json', type=str, default=None,
                        help="Path to best config JSON file (default: out_dir/best_she.json)")

    # Main.py location
    parser.add_argument('--main_script', type=str, default='src/main.py',
                        help="Path to main.py script")
    parser.add_argument('--data_path', type=str, default='./data',
                        help="Path to data directory")

    args = parser.parse_args()

    if args.summary_csv is None:
        args.summary_csv = os.path.join(args.out_dir, 'summary_she.csv')
    if args.best_json is None:
        args.best_json = os.path.join(args.out_dir, 'best_she.json')

    return args


def compute_she_k(r: int, k_factor: float) -> int:
    if k_factor == 1.0:
        return max(r, 64)
    else:
        return math.ceil(k_factor * r)


def build_search_space(args) -> List[Dict]:
    configs = []

    for r in args.r_list:
        for kf in args.k_factor_list:
            she_k = compute_she_k(r, kf)
            for T in args.T_list:
                for alpha in args.alpha_list:
                    config = {
                        'r': r,
                        'k': she_k,
                        'T': T,
                        'alpha': alpha,
                    }
                    configs.append(config)

    return configs


def get_config_id(config: Dict) -> str:
    a_str = str(config['alpha']).replace('.', 'p')
    return f"r{config['r']}_k{config['k']}_T{config['T']}_a{a_str}"


def get_result_json_path(args, config: Dict) -> str:
    config_id = get_config_id(config)
    filename = f"results_SHE_{config_id}.json"
    return os.path.join(args.out_dir, filename)


def get_log_path(args, config: Dict) -> str:
    os.makedirs(os.path.join(args.out_dir, 'logs'), exist_ok=True)
    config_id = get_config_id(config)
    return os.path.join(args.out_dir, 'logs', f"she_{config_id}.log")


def run_config(args, config: Dict) -> Tuple[bool, Optional[Dict], float]:
    config_id = get_config_id(config)
    final_result_json_path = get_result_json_path(args, config)
    log_path = get_log_path(args, config)

    if args.resume and os.path.exists(final_result_json_path):
        print(f"  Skipping {config_id} (result exists)")
        try:
            with open(final_result_json_path, 'r') as f:
                results = json.load(f)
            return True, results, 0.0
        except Exception as e:
            print(f"  Warning: Failed to load existing result: {e}")

    # Construct command
    cmd = [
        sys.executable,
        args.main_script,
        '--ppi_dataset', args.dataset,
        '--model', args.model,
        '--she_r', str(config['r']),
        '--she_k', str(config['k']),
        '--she_T', str(config['T']),
        '--she_alpha', str(config['alpha']),
        '--n_repeats', str(args.n_repeats),
        '--output_dir', args.out_dir,
        '--data_path', args.data_path
    ]

    print(f"  Running command...")

    start_time = time.time()
    try:
        with open(log_path, 'w') as log_file:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=3600
            )
        elapsed_time = time.time() - start_time

        if result.returncode != 0:
            print(f"  Error: Command failed with return code {result.returncode}")
            # Print last 20 lines of log file for debugging
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    print(f"  Log excerpt (last {min(20, len(lines))} lines):")
                    for line in lines[-20:]:
                        print(f"    {line.rstrip()}")
            except:
                pass
            return False, None, elapsed_time

        # main.py saves to: results_{ppi_dataset}_{model}.json
        default_result_path = os.path.join(args.out_dir, f"results_{args.dataset}_SHE.json")

        # Read and rename result JSON
        if os.path.exists(default_result_path):
            with open(default_result_path, 'r') as f:
                results = json.load(f)

            # Rename to include config_id
            os.rename(default_result_path, final_result_json_path)

            return True, results, elapsed_time
        else:
            print(f"  Error: Result JSON not found at {default_result_path}")
            return False, None, elapsed_time

    except subprocess.TimeoutExpired:
        print(f"  Error: Command timeout (>3600s)")
        return False, None, time.time() - start_time
    except Exception as e:
        print(f"  Error: {e}")
        return False, None, time.time() - start_time


def parse_metrics(results: Dict) -> Dict:
    metrics = {}

    if results is None:
        return metrics

    # Binary metrics
    binary = results.get('binary_metrics', {})
    metrics['auprc'] = binary.get('auprc', 0.0)
    metrics['auprc_se'] = binary.get('auprc_se', 0.0)
    metrics['auroc'] = binary.get('auroc', 0.0)
    metrics['auroc_se'] = binary.get('auroc_se', 0.0)
    metrics['f1'] = binary.get('f1', 0.0)
    metrics['f1_se'] = binary.get('f1_se', 0.0)
    metrics['precision'] = binary.get('precision', 0.0)
    metrics['recall'] = binary.get('recall', 0.0)
    metrics['accuracy'] = binary.get('accuracy', 0.0)
    metrics['threshold'] = binary.get('threshold', 0.0)
    metrics['threshold_se'] = binary.get('threshold_se', 0.0)

    # Complex metrics
    complex_metrics = results.get('complex_metrics', {})
    metrics['complex_f1'] = complex_metrics.get('f1', 0.0)
    metrics['complex_f1_se'] = complex_metrics.get('f1_se', 0.0)
    metrics['complex_acc'] = complex_metrics.get('acc', 0.0)
    metrics['complex_sn'] = complex_metrics.get('sn', 0.0)
    metrics['complex_ppv'] = complex_metrics.get('ppv', 0.0)

    return metrics


def append_to_csv(args, config: Dict, metrics: Dict, elapsed_time: float, success: bool):
    csv_path = args.summary_csv

    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)

    row = {
        'config_id': get_config_id(config),
        'r': config['r'],
        'k': config['k'],
        'T': config['T'],
        'alpha': config['alpha'],
        'success': success,
        'elapsed_time': elapsed_time,
    }
    row.update(metrics)

    fieldnames = list(row.keys())

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_csv_results(csv_path: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        return []

    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    return results


def save_best_config(args, csv_path: str):
    results = load_csv_results(csv_path)

    if not results:
        print("No results found")
        return

    # Filter successful runs
    successful = [r for r in results if r.get('success', 'False') == 'True']

    if not successful:
        print("No successful runs found")
        return

    # Sort by AUPRC (descending), then by complex_f1
    successful.sort(
        key=lambda x: (
            -float(x.get('auprc', 0)),
            -float(x.get('complex_f1', 0))
        )
    )

    # Get top-N configs
    top_n = 3
    print(f"\n{'='*80}")
    print(f"Top {top_n} configurations (sorted by AUPRC, then Complex F1):")
    print(f"{'='*80}")

    for i, result in enumerate(successful[:top_n], 1):
        print(f"\nRank {i}: {result['config_id']}")
        print(f"  r={result['r']}, k={result['k']}, T={result['T']}, alpha={result['alpha']}")
        print(f"  AUPRC: {float(result['auprc']):.4f} 卤 {float(result.get('auprc_se', 0)):.4f}")
        print(f"  AUROC: {float(result['auroc']):.4f} 卤 {float(result.get('auroc_se', 0)):.4f}")
        print(f"  Binary F1: {float(result['f1']):.4f} 卤 {float(result.get('f1_se', 0)):.4f}")
        print(f"  Complex F1: {float(result['complex_f1']):.4f} 卤 {float(result.get('complex_f1_se', 0)):.4f}")
        print(f"  Complex Acc: {float(result['complex_acc']):.4f}")
        print(f"  Threshold: {float(result['threshold']):.4f} 卤 {float(result.get('threshold_se', 0)):.4f}")
        print(f"  Time: {float(result['elapsed_time']):.1f}s")

    # Save best config to JSON
    best = successful[0]
    best_config = {
        'config_id': best['config_id'],
        'parameters': {
            'r': int(best['r']),
            'k': int(best['k']),
            'T': int(best['T']),
            'alpha': float(best['alpha'])
        },
        'metrics': {
            'auprc': float(best['auprc']),
            'auprc_se': float(best.get('auprc_se', 0)),
            'auroc': float(best['auroc']),
            'auroc_se': float(best.get('auroc_se', 0)),
            'binary_f1': float(best['f1']),
            'binary_f1_se': float(best.get('f1_se', 0)),
            'complex_f1': float(best['complex_f1']),
            'complex_f1_se': float(best.get('complex_f1_se', 0)),
            'complex_acc': float(best['complex_acc']),
            'complex_sn': float(best['complex_sn']),
            'complex_ppv': float(best['complex_ppv']),
            'threshold': float(best['threshold']),
            'threshold_se': float(best.get('threshold_se', 0))
        },
        'elapsed_time': float(best['elapsed_time'])
    }

    with open(args.best_json, 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"\nBest configuration saved to: {args.best_json}")


def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print(f"SHE Grid Search")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Search space:")
    print(f"  r_list: {args.r_list}")
    print(f"  k_factor_list: {args.k_factor_list}")
    print(f"  T_list: {args.T_list}")
    print(f"  alpha_list: {args.alpha_list}")
    print(f"Evaluation:")
    print(f"  n_repeats: {args.n_repeats}")
    print(f"  folds: {args.folds}")
    print(f"  seed: {args.seed}")
    print(f"Output:")
    print(f"  Directory: {args.out_dir}")
    print(f"  Summary CSV: {args.summary_csv}")
    print(f"  Best JSON: {args.best_json}")
    print(f"{'='*80}\n")

    # Build search space
    configs = build_search_space(args)
    print(f"Total configurations: {len(configs)}\n")

    # Create output directories
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'logs'), exist_ok=True)

    # Run each configuration
    for i, config in enumerate(configs, 1):
        config_id = get_config_id(config)
        print(f"\n[{i}/{len(configs)}] Configuration: {config_id}")
        print(f"  r={config['r']}, k={config['k']}, T={config['T']}, alpha={config['alpha']}")

        success, results, elapsed_time = run_config(args, config)

        if success:
            metrics = parse_metrics(results)
            print(f"  Success! AUPRC: {metrics['auprc']:.4f}, Complex F1: {metrics['complex_f1']:.4f}, Time: {elapsed_time:.1f}s")
        else:
            metrics = {}
            print(f"  Failed! Time: {elapsed_time:.1f}s")

        append_to_csv(args, config, metrics, elapsed_time, success)

    print(f"\n{'='*80}")
    print("Grid search completed!")
    print(f"{'='*80}")
    print(f"Results saved to: {args.summary_csv}")

    # Save best configuration
    save_best_config(args, args.summary_csv)


if __name__ == "__main__":
    main()



