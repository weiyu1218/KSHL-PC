import argparse
import os
import json
import pandas as pd


def load_metrics(metrics_json_path):
    if not os.path.exists(metrics_json_path):
        return None
    with open(metrics_json_path, 'r') as f:
        return json.load(f)


def summarize_nodecls_results(results_base_dir, aspects=['BP', 'MF'],
                               embedding_types=['FUSED', 'node2vec'],
                               out_csv=None):

    summary_data = []

    for emb_type in embedding_types:
        for aspect in aspects:
            metrics_path = os.path.join(results_base_dir, emb_type, aspect, 'metrics.json')
            metrics = load_metrics(metrics_path)

            if metrics is None:
                print(f"Warning: No metrics found for {emb_type}/{aspect}")
                row = {
                    'Embedding': emb_type,
                    'Aspect': aspect,
                    'Micro-F1': None,
                    'Macro-F1': None,
                    'Micro-AUPRC': None,
                    'Std(Micro-F1)': None,
                    'Std(Macro-F1)': None,
                    'Std(Micro-AUPRC)': None
                }
            else:
                row = {
                    'Embedding': emb_type,
                    'Aspect': aspect,
                    'Micro-F1': metrics.get('micro_f1_mean', None),
                    'Macro-F1': metrics.get('macro_f1_mean', None),
                    'Micro-AUPRC': metrics.get('micro_auprc_mean', None),
                    'Std(Micro-F1)': metrics.get('micro_f1_std', None),
                    'Std(Macro-F1)': metrics.get('macro_f1_std', None),
                    'Std(Micro-AUPRC)': metrics.get('micro_auprc_std', None)
                }

            summary_data.append(row)

    df = pd.DataFrame(summary_data)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
        df.to_csv(out_csv, index=False, float_format='%.4f')
        print(f"Summary table saved to: {out_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Summarize multi-label node classification results')
    parser.add_argument('--results_dir', required=True,
                        help='Base directory containing results (e.g., results/case_tasks/node_cls)')
    parser.add_argument('--aspects', nargs='+', default=['BP', 'MF'],
                        help='GO aspects to summarize')
    parser.add_argument('--embeddings', nargs='+', default=['FUSED', 'node2vec'],
                        help='Embedding types to compare')
    parser.add_argument('--out_csv', required=True,
                        help='Output CSV path for summary table')
    args = parser.parse_args()

    print(f"\nSummarizing node classification results...")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Aspects: {args.aspects}")
    print(f"  Embeddings: {args.embeddings}")

    df = summarize_nodecls_results(
        args.results_dir,
        aspects=args.aspects,
        embedding_types=args.embeddings,
        out_csv=args.out_csv
    )

    print(f"\nSummary Table:")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()



