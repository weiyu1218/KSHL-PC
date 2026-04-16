#!/usr/bin/env python
"""
杩愯鎵€鏈夊熀绾挎ā鍨嬬敓鎴恊mbeddings
"""
import argparse
import subprocess
import os


BASELINE_MODELS = {
    'GAT': 'gat_embed.py',
    'GCN': 'gcn_embed.py',
    'GIN': 'gin_embed.py',
    'GAE': 'gae_embed.py',
    'GraphMAE': 'graphmae_embed.py'
}


def run_baseline(model_name, script_name, data_root, out_dir, epochs, device, verbose):
    """
    杩愯鍗曚釜鍩虹嚎妯″瀷
    """
    script_path = os.path.join('experiments', 'node_cls', 'baselines', script_name)
    out_path = os.path.join(out_dir, f'{model_name.lower()}.pt')

    print(f"\n{'='*60}")
    print(f"Training {model_name} model")
    print(f"{'='*60}")

    cmd = [
        'python', script_path,
        '--data_root', data_root,
        '--out_path', out_path,
        '--epochs', str(epochs),
        '--device', device
    ]

    if verbose:
        cmd.append('--verbose')

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"Error: {model_name} training failed")
        return False

    print(f"\n{model_name} embeddings saved to: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for all baseline models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

1. Generate all baseline embeddings:
   python run_baselines.py --data_root data/Saccharomyces_cerevisiae --out_dir results/emb/baselines

2. Generate specific models:
   python run_baselines.py --models GAT GCN --out_dir results/emb/baselines

3. Use GPU and verbose output:
   python run_baselines.py --out_dir results/emb/baselines --device cuda --verbose
        """
    )

    parser.add_argument('--data_root', default='data/Saccharomyces_cerevisiae',
                        help='Data root directory')
    parser.add_argument('--out_dir', default='results/emb/baselines',
                        help='Output directory for embeddings')
    parser.add_argument('--models', nargs='+', choices=list(BASELINE_MODELS.keys()),
                        default=list(BASELINE_MODELS.keys()),
                        help='Baseline models to train (default: all)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print training progress')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Generating Baseline Model Embeddings")
    print(f"{'='*60}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.out_dir}")
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")

    success_count = 0
    for model_name in args.models:
        script_name = BASELINE_MODELS[model_name]
        success = run_baseline(
            model_name,
            script_name,
            args.data_root,
            args.out_dir,
            args.epochs,
            args.device,
            args.verbose
        )
        if success:
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(args.models)} models")
    print(f"{'='*60}")
    print(f"Embeddings saved to: {args.out_dir}")


if __name__ == '__main__':
    main()



