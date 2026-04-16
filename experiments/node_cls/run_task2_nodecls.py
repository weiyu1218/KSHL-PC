#!/usr/bin/env python
import argparse
import os
import sys
import subprocess


def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error: {description} failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run Task 2: GO-BP/MF Multi-label Node Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

1. Build labels only:
   python run_task2_nodecls.py --dataset Mann --build_labels_only

2. Run classification with existing labels:
   python run_task2_nodecls.py --dataset Mann --embedding_type FUSED --skip_build_labels

3. Full pipeline (build labels + classify):
   python run_task2_nodecls.py --dataset Mann --embedding_type FUSED

4. Compare multiple embeddings:
   python run_task2_nodecls.py --dataset Mann --embedding_type FUSED node2vec
        """
    )

    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., Mann)')
    parser.add_argument('--data_root', default='data/Saccharomyces_cerevisiae',
                        help='Data root directory')
    parser.add_argument('--results_root', default='results',
                        help='Results root directory')

    parser.add_argument('--build_labels_only', action='store_true',
                        help='Only build labels and exit')
    parser.add_argument('--skip_build_labels', action='store_true',
                        help='Skip building labels (use existing)')

    parser.add_argument('--embedding_type', nargs='+',
                        default=['FUSED'],
                        help='Embedding types to use (FUSED, node2vec, GAT, GCN, GIN, GAE, GraphMAE)')
    parser.add_argument('--embedding_dir', default='results/emb',
                        help='Directory containing embeddings')

    parser.add_argument('--aspects', nargs='+', default=['BP', 'MF'],
                        help='GO aspects to classify')
    parser.add_argument('--min_freq', type=int, default=30,
                        help='Minimum term frequency')

    parser.add_argument('--clf', choices=['lr', 'mlp'], default='mlp',
                        help='Classifier type')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--mlp_hidden', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--mlp_lr', type=float, default=0.001)
    parser.add_argument('--mlp_epochs', type=int, default=100)
    parser.add_argument('--mlp_batch_size', type=int, default=64)
    parser.add_argument('--device', default='cpu')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    base_dir = os.getcwd()
    go_slim_path = os.path.join(base_dir, args.data_root, 'GO', 'go_slim_mapping.tab.txt')
    protein_list = os.path.join(base_dir, args.data_root, 'Gene_Entry_ID_list', 'Protein_list.csv')
    labels_dir = os.path.join(base_dir, args.results_root, 'labels')
    case_tasks_dir = os.path.join(base_dir, args.results_root, 'case_tasks', 'node_cls')
    tables_dir = os.path.join(base_dir, args.results_root, 'tables')

    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(case_tasks_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Task 2: GO-BP/MF Multi-label Node Classification")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Aspects: {args.aspects}")
    print(f"Embedding types: {args.embedding_type}")
    print(f"Classifier: {args.clf}")
    print(f"Cross-validation: {args.folds}-Fold 脳 {args.repeats} repeats")

    if not args.skip_build_labels:
        for aspect in args.aspects:
            labels_npz = os.path.join(labels_dir, f'go_{aspect.lower()}.npz')

            cmd = [
                'python', 'experiments/node_cls/labels_go.py',
                '--aspect', aspect,
                '--go_slim_path', go_slim_path,
                '--protein_list', protein_list,
                '--min_freq', str(args.min_freq),
                '--out_npz', labels_npz
            ]

            success = run_command(cmd, f"Building GO-{aspect} labels")
            if not success:
                print(f"Failed to build labels for {aspect}")
                return

    if args.build_labels_only:
        print("\nLabels built successfully. Exiting (--build_labels_only specified).")
        return

    embedding_paths = {}
    for emb_type in args.embedding_type:
        if emb_type == 'FUSED':
            emb_file = 'fused.pt'
        elif emb_type == 'node2vec':
            emb_file = 'node2vec.pt'
        elif emb_type in ['GAT', 'GCN', 'GIN', 'GAE', 'GraphMAE']:
            emb_file = f'baselines/{emb_type.lower()}.pt'
        else:
            print(f"Warning: Unknown embedding type {emb_type}, skipping")
            continue

        emb_path = os.path.join(base_dir, args.embedding_dir, emb_file)
        if not os.path.exists(emb_path):
            print(f"Warning: Embedding not found: {emb_path}")
            continue

        embedding_paths[emb_type] = emb_path

    if not embedding_paths:
        print("Error: No valid embeddings found")
        return

    for emb_type, emb_path in embedding_paths.items():
        for aspect in args.aspects:
            labels_npz = os.path.join(labels_dir, f'go_{aspect.lower()}.npz')
            out_dir = os.path.join(case_tasks_dir, emb_type, aspect)

            cmd = [
                'python', 'experiments/node_cls/tasks_node_cls.py',
                '--aspect', aspect,
                '--labels_npz', labels_npz,
                '--embedding', emb_path,
                '--folds', str(args.folds),
                '--repeats', str(args.repeats),
                '--seed', str(args.seed),
                '--clf', args.clf,
                '--dropout', str(args.dropout),
                '--device', args.device,
                '--out_dir', out_dir
            ]

            if args.clf == 'mlp':
                cmd.extend(['--mlp_hidden'] + [str(h) for h in args.mlp_hidden])
                cmd.extend(['--mlp_lr', str(args.mlp_lr)])
                cmd.extend(['--mlp_epochs', str(args.mlp_epochs)])
                cmd.extend(['--mlp_batch_size', str(args.mlp_batch_size)])

            if args.verbose:
                cmd.append('--verbose')

            success = run_command(cmd, f"Training {emb_type} on GO-{aspect}")
            if not success:
                print(f"Warning: Training failed for {emb_type}/{aspect}")

    summary_csv = os.path.join(tables_dir, 'nodecls_summary.csv')
    cmd = [
        'python', 'experiments/node_cls/summarize_nodecls.py',
        '--results_dir', case_tasks_dir,
        '--aspects'] + args.aspects + [
        '--embeddings'] + list(embedding_paths.keys()) + [
        '--out_csv', summary_csv
    ]

    success = run_command(cmd, "Generating summary table")
    if not success:
        print("Warning: Failed to generate summary table")

    print(f"\n{'='*60}")
    print("Task 2 completed!")
    print(f"{'='*60}")
    print(f"Results saved to: {case_tasks_dir}")
    print(f"Summary table: {summary_csv}")
    print(f"\nTo view summary:")
    print(f"  cat {summary_csv}")


if __name__ == '__main__':
    main()



