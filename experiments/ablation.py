import os
import sys
import json
import argparse
from types import SimpleNamespace
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation runner for KSHL-PC removal-and-retrain experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset and base SHE params (fixed across ablations)
    parser.add_argument('--dataset', type=str, default='Mann', choices=['DIP', 'BioGRID', 'Mann'],
                        help='PPI dataset to use')
    parser.add_argument('--species', type=str, default='Saccharomyces_cerevisiae',
                        help='Species directory name')
    parser.add_argument('--data_path', type=str, default='data', help='Path to data')

    parser.add_argument('--she_r', type=int, default=None, help='SHE SVD rank r')
    parser.add_argument('--she_k', type=int, default=None, help='SHE embedding dim k')
    parser.add_argument('--she_T', type=int, default=10, help='SHE window size T')
    parser.add_argument('--she_alpha', type=float, default=0.1, help='SHE alpha')
    parser.add_argument('--load_best_from', type=str, default=None,
                        help='Optional path to best_she.json (overrides r/k/T/alpha)')

    # Training and evaluation
    parser.add_argument('--n_repeats', type=int, default=30, help='Num repeats for negative sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs recorded in the result config')

    # Output
    parser.add_argument('--out_root', type=str, default='results/ablation',
                        help='Root output dir for ablations')
    parser.add_argument('--resume', action='store_true', help='Skip ablations with existing results')

    # Prediction export parameters
    parser.add_argument('--export_preds', action='store_true', default=True , help='Export predictions to CSV')
    parser.add_argument('--export_dir', type=str, default=None, help='Directory for exported predictions')
    parser.add_argument('--export_val', action='store_true', default= True , help='Also export validation set predictions')

    # Knowledge fusion parameters
    parser.add_argument('--fusion_method', type=str, default='ae', choices=['ae', 'concat'],
                        help='Fusion method')
    parser.add_argument('--fusion_dim', type=int, default=128,
                        help='Dimension of fused embeddings')
    parser.add_argument('--fusion_epochs', type=int, default=100,
                        help='Training epochs for autoencoder fusion')
    parser.add_argument('--fusion_lr', type=float, default=1e-3,
                        help='Learning rate for autoencoder fusion')
    parser.add_argument('--keft_T', type=int, default=12,
                        help='Number of temporal windows for activity features')
    parser.add_argument('--keft_k', type=int, default=1,
                        help='Threshold coefficient for temporal activity detection')

    # Selection of ablations (only the essential set)
    # w_o_spectral: non-spectral structural features only (with KEFT fusion)
    # w_o_spectral_no_kg: non-spectral structural features only (without KEFT fusion)
    parser.add_argument('--ablations', type=str, nargs='*',
                        default=['full', 'w_o_keft', 'w_o_pin', 'w_o_spectral'],
                        help='Ablations: full, w_o_keft, w_o_pin, w_o_spectral, w_o_spectral_no_kg')

    return parser.parse_args()


def build_args(base, overrides=None):
    """Build args Namespace for main.main using parser defaults and overrides."""
    if overrides is None:
        overrides = {}

    # SHE parameters from grid search results
    # Defaults mirror src/main.py parser
    cfg = {
        'species': base.species,
        'data_path': base.data_path,
        'feature_path': 'protein_feature',
        'PPI_path': 'PPI',
        'PC_path': 'protein_complex',
        'model': 'SHE',
        'ppi_dataset': base.dataset,
        'lr': 0.001,
        'hidden1': 200,
        'hidden2': 100,
        'droprate': 0.5,
        'epochs': base.epochs,
        'output_dir': '',  # to be set by caller
        'save_model': False,
        'she_r': base.she_r if base.she_r is not None else 64,
        'she_k': base.she_k if base.she_k is not None else 64,
        'she_T': base.she_T,
        'she_alpha': base.she_alpha,
        'use_keft_fusion': True,
        'fusion_method': getattr(base, 'fusion_method', 'ae'),
        'fusion_dim': getattr(base, 'fusion_dim', 128),
        'fusion_epochs': getattr(base, 'fusion_epochs', 100),
        'fusion_lr': getattr(base, 'fusion_lr', 1e-3),
        'keft_T': getattr(base, 'keft_T', 12),
        'keft_k': getattr(base, 'keft_k', 1),
        'keft_cache': True,
        'seed': base.seed,
        'n_repeats': base.n_repeats,
    }

    cfg.update(overrides)

    # Add export parameters with default values
    cfg['export_preds'] = getattr(base, 'export_preds', True)
    cfg['export_val'] = getattr(base, 'export_val', True)
    cfg['export_dir'] = getattr(base, 'export_dir', None)
    cfg['variant'] = getattr(base, 'variant', None)

    return SimpleNamespace(**cfg)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def run_one(ablation_name, base_args, overrides):
    # Import main.py from src folder
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(here, '..', 'src')
    src_dir = os.path.abspath(src_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from main import main as run_main

    out_dir = os.path.join(base_args.out_root, ablation_name)
    ensure_dir(out_dir)

    # Determine model name for expected result file
    model_name = overrides.get('model', 'SHE')
    # main.py writes to: results_{dataset}_{model}.json inside output_dir
    result_json = os.path.join(out_dir, f"results_{base_args.dataset}_{model_name}.json")
    if base_args.resume and os.path.exists(result_json):
        print(f"[skip] {ablation_name}: results exist: {result_json}")
        return True, result_json

    args = build_args(base_args, overrides)
    args.output_dir = out_dir

    # Set variant name for exports
    if not hasattr(args, 'variant') or args.variant is None:
        args.variant = ablation_name

    print(f"\n=== Ablation: {ablation_name} ===")
    print(f"Output: {out_dir}")
    print(f"Model: {model_name}")
    print(f"Config: r={args.she_r}, k={args.she_k}, T={args.she_T}, alpha={args.she_alpha}")
    print(f"KEFT fusion: {args.use_keft_fusion}")
    if hasattr(args, 'export_preds') and args.export_preds:
        print(f"Export predictions: enabled (variant={args.variant})")

    try:
        run_main(args)
    except SystemExit:
        # In case downstream uses argparse that calls sys.exit
        pass
    except Exception as e:
        print(f"[error] {ablation_name} failed: {e}")
        return False, result_json

    ok = os.path.exists(result_json)
    if not ok:
        print(f"[warn] Result JSON not found: {result_json}")
    return ok, result_json


def parse_metrics(path):
    try:
        with open(path, 'r') as f:
            obj = json.load(f)
        m = obj.get('binary_metrics', {})
        c = obj.get('complex_metrics', {})
        return {
            'auprc': m.get('auprc', None),
            'auroc': m.get('auroc', None),
            'f1': m.get('f1', None),
            'precision': m.get('precision', None),
            'recall': m.get('recall', None),
            'complex_f1': c.get('f1', None),
            'complex_acc': c.get('acc', None),
            'complex_sn': c.get('sn', None),
            'complex_ppv': c.get('ppv', None),
        }
    except Exception:
        return {}


def write_summary(dataset, out_root, records):
    import csv
    summary_dir = ensure_dir(out_root)
    summary_csv = os.path.join(summary_dir, 'summary_ablation.csv')
    fieldnames = [
        'ablation', 'result_path', 'success',
        'auprc', 'auroc', 'f1', 'precision', 'recall',
        'complex_f1', 'complex_acc', 'complex_sn', 'complex_ppv',
        'timestamp'
    ]
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            w.writerow(rec)
    print(f"\nSummary saved to: {summary_csv}")


def main():
    base = parse_args()

    # Save user-specified CLI parameters before loading from JSON
    user_specified = {
        'r': base.she_r,
        'k': base.she_k,
        'T': base.she_T,
        'alpha': base.she_alpha,
    }

    # Optionally load best SHE params from a JSON produced by grid_she
    # Only use JSON values for parameters NOT specified by user
    if base.load_best_from and os.path.exists(base.load_best_from):
        try:
            with open(base.load_best_from, 'r') as f:
                best = json.load(f)
            params = best.get('parameters', {})
            # Only override if user did not specify the parameter
            if user_specified['r'] is None:
                base.she_r = int(params.get('r', base.she_r))
            if user_specified['k'] is None:
                base.she_k = int(params.get('k', base.she_k))
            # T and alpha already have defaults, only override if from JSON
            if user_specified['T'] == 10:  # default value
                base.she_T = int(params.get('T', base.she_T))
            if user_specified['alpha'] == 0.1:  # default value
                base.she_alpha = float(params.get('alpha', base.she_alpha))
            print(f"Loaded SHE params from {base.load_best_from}: {params}")
            print(f"User-specified params (will override JSON): r={user_specified['r']}, k={user_specified['k']}")
        except Exception as e:
            print(f"[warn] Failed to load best config from {base.load_best_from}: {e}")

    # Ablation experiments configuration
    # 1) full: SHE + KEFT
    # 2) w_o_keft: SHE only
    # 3) w_o_pin: no PPI network (use CT features only)
    # 4) w_o_spectral: non-spectral structural features with KEFT fusion (aligned dimension)
    # 5) w_o_spectral_no_kg: non-spectral structural features without KEFT fusion
    ablation_map = {
        'full': {},
        'w_o_keft': {'use_keft_fusion': False},
        'w_o_pin': {'model': 'None'},
        'w_o_spectral': {'model': 'StructStats'},
        'w_o_spectral_no_kg': {'model': 'StructStats', 'use_keft_fusion': False},
    }

    records = []

    for ab in base.ablations:
        if ab not in ablation_map:
            print(f"[warn] Unknown ablation '{ab}', skip")
            continue

        ok, path = run_one(ab, base, ablation_map[ab])
        metrics = parse_metrics(path) if ok else {}
        records.append({
            'ablation': ab,
            'result_path': path,
            'success': ok,
            **metrics,
            'timestamp': str(datetime.now()),
        })

    write_summary(base.dataset, base.out_root, records)


if __name__ == '__main__':
    main()



