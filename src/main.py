import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))


def main(args):
    import json
    import pickle
    from datetime import datetime

    import networkx as nx
    import numpy as np
    import pandas as pd
    import torch
    from pandas.core.frame import DataFrame

    from Train_PC import HGC_DNN
    from utils import (
        Nested_list_dup,
        convert_ppi,
        count_unique_elements,
        load_txt_list,
        preprocessing_PPI,
        sequence_CT,
        try_gpu,
    )

    # Set random seeds for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {args.seed}")

    # Load protein sequence metadata (without CT encoding for now)
    Sequence_path = os.path.join(args.data_path, args.species, args.feature_path,
                                "uniprot-sequences-2023.05.10-01.31.31.11.tsv")
    Sequence = pd.read_csv(Sequence_path,sep='\t')

    # Only compute CT features if needed by the model
    needs_ct = args.model == 'None'
    if needs_ct:
        print("Computing CT features (required for model)...")
        Sequence_feature = sequence_CT(Sequence)
    else:
        print("Skipping CT feature computation (not required for model)")
        Sequence_feature = Sequence

    # Load PPI dataset
    ppi_dataset = args.ppi_dataset.upper()

    if ppi_dataset == 'MANN':
        # Mann-PPI uses special CSV format
        PPI = os.path.join(args.data_path, args.species, args.PPI_path, "Mann_PPI.csv")
        PPI = pd.read_csv(PPI, sep=";")
        PPI = (
            PPI.assign(target=PPI['target'].str.split(';'))
            .explode('target')
            .reset_index(drop=True)
            [['source', 'target']]
            .query('source != target')
        )
        PPI_trans = PPI[['target', 'source']].copy()
        PPI_trans.columns = ['protein1', 'protein2']
        PPI.columns = ['protein1', 'protein2']
        PPI = pd.concat([PPI, PPI_trans], axis=0).reset_index(drop=True)
    elif ppi_dataset in ['DIP', 'BIOGRID']:
        # DIP and BioGRID use system name format from AdaPPI_Dataset
        ppi_file = f"{ppi_dataset.lower()}.txt"
        PPI_path = os.path.join(args.data_path, args.species, args.PPI_path, "AdaPPI_Dataset", ppi_dataset, ppi_file)
        PPI = pd.read_csv(PPI_path, sep='\t', header=None, names=['protein1', 'protein2'])
        # Add reverse edges
        PPI_trans = PPI[['protein2', 'protein1']].copy()
        PPI_trans.columns = ['protein1', 'protein2']
        PPI = pd.concat([PPI, PPI_trans], axis=0).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown PPI dataset: {ppi_dataset}. Choose from: DIP, BioGRID, Mann")

    PPI,Protein_dict = preprocessing_PPI(PPI,Sequence_feature)
    PPI.to_csv(os.path.join(args.data_path, args.species, args.PPI_path,
                                "ID_Change_PPI.txt"),
               index=False, header=False,sep="\t")

    Protein_dict.to_csv(os.path.join(args.data_path, args.species,
                            "Gene_Entry_ID_list/Protein_list.csv"),
               index=False, header=False, sep="\t")

    PPI_list = PPI.values.tolist()
    PPI_list = Nested_list_dup(PPI_list)

    # Construct PPI hypergraph
    G = nx.Graph()
    G.add_edges_from(PPI_list)
    PPI_hyperedge_dup = list(nx.find_cliques(G))
    unique_elements = count_unique_elements(PPI_hyperedge_dup)

    all_hyperedges = PPI_hyperedge_dup.copy()
    print(f"PPI cliques: {len(PPI_hyperedge_dup)} hyperedges")

    edge_list_data = {}
    edge_list_data["num_vertices"] = len(unique_elements)
    edge_list_data["PPI_edge_list"] = PPI_list
    edge_list_data["PPI_cliques_list"] = all_hyperedges

    f_save = open(os.path.join(args.data_path, args.species, args.PPI_path,
                                "'PPI_cliques_Hyperedge.pkl'"), 'wb')
    pickle.dump(edge_list_data, f_save)
    f_save.close()

    # Generate CT feature tensor only if needed
    X = None
    if needs_ct:
        Sequence_feature_merged = pd.merge(Protein_dict, Sequence_feature, how='inner')
        Sequence_feature_merged = Sequence_feature_merged.sort_values(by=['ID'])
        Sequence_feature_tensor = DataFrame(Sequence_feature_merged['features_seq'].to_list())
        X = torch.FloatTensor(np.array(Sequence_feature_tensor))
        X = X.to(device=try_gpu())
        CT_Embedding_path = os.path.join(args.data_path, args.species, args.feature_path,
                                    "protein_feature_CT.pt")
        torch.save(X, CT_Embedding_path)
        print(f"CT features saved to: {CT_Embedding_path}")

    # Generate feature embeddings
    Embedding_path = os.path.join(args.data_path, args.species, args.feature_path,
                                    "protein_feature_SHE.pt")

    if args.model == 'Node2vec':
        from node2vec import Node2Vec

        graph = pd.read_csv(os.path.join(args.data_path, args.species, args.PPI_path,
                                "ID_Change_PPI.txt"), sep='\t', header=None)
        edgelist = graph.values.tolist()
        G = nx.from_edgelist(edgelist)
        model = Node2Vec(G, dimensions=64, walk_length=80, num_walks=10, p=8, q=1, workers=1)
        model = model.fit(window=10, min_count=1, batch_words=4)
        embedding = model.wv.vectors
        embedding = pd.DataFrame(embedding)
        Embedding = torch.FloatTensor(np.array(embedding)).to(device=try_gpu())
    elif args.model == 'SHE':
        from she_embed import compute_she_embedding

        r = args.she_r if args.she_r is not None else 64
        k = args.she_k if args.she_k is not None else 64
        T_eff = args.she_T
        alpha_eff = args.she_alpha

        print(f"Running SHE with r={r}, k={k}, T={T_eff}, alpha={alpha_eff}, seed={args.seed}")

        Z_v, elapsed_time = compute_she_embedding(
            edge_list_data, None,
            r=r,
            k=k,
            T=T_eff,
            alpha=alpha_eff,
            seed=args.seed,
            vol_type=args.vol_type
        )
        Embedding = Z_v.to(device=try_gpu())
        print(f"SHE embedding completed in {elapsed_time:.2f}s, shape: {Embedding.shape}")
    elif args.model == 'StructStats':
        from struct_stats import compute_structstats_embedding

        k = args.she_k if args.she_k is not None else 64

        print(f"Running StructStats with target_dim={k}, seed={args.seed}")
        print("Note: she_r, she_T, she_alpha are not applicable for StructStats")

        try:
            Embedding = compute_structstats_embedding(
                G=G,
                cliques=PPI_hyperedge_dup,
                num_vertices=len(unique_elements),
                target_dim=k,
                seed=args.seed
            )
            Embedding = Embedding.to(device=try_gpu())
            print(f"StructStats embedding completed, shape: {Embedding.shape}")
        except Exception as e:
            print(f"ERROR in StructStats embedding: {e}")
            import traceback
            traceback.print_exc()
            raise
    elif args.model == 'None':
        Embedding = X
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose from: Node2vec, SHE, StructStats, None")

    # Save embedding
    torch.save(Embedding, Embedding_path)
    print(f"Embedding saved to: {Embedding_path}")

    if args.use_keft_fusion:
        print(f"\n{'='*60}")
        print("Knowledge-Enhanced Feature Fusion")
        print(f"{'='*60}\n")

        go_slim_path = os.path.join(args.data_path, args.species, "GO", "go_slim_mapping.tab.txt")
        series_path = os.path.join(args.data_path, args.species, "expression", "series_matrix.txt")

        if not os.path.exists(go_slim_path):
            print(f"WARNING: GO slim file not found at {go_slim_path}. Skipping knowledge fusion.")
        elif not os.path.exists(series_path):
            print(f"WARNING: Series matrix file not found at {series_path}. Skipping knowledge fusion.")
        else:
            try:
                Z_know, know_metadata = build_knowledge_features(
                    protein_dict=Protein_dict,
                    go_slim_path=go_slim_path,
                    series_path=series_path,
                    T=args.keft_T,
                    k=args.keft_k,
                    normalize=True,
                    use_cache=args.keft_cache
                )

                Z_fused = fuse_embeddings(
                    she_emb=Embedding,
                    know_features=Z_know,
                    method=args.fusion_method,
                    out_dim=args.fusion_dim,
                    epochs=args.fusion_epochs,
                    lr=args.fusion_lr,
                    verbose=True
                )

                fused_path = os.path.join(args.data_path, args.species, args.feature_path,
                                          "protein_feature_fused.pt")
                torch.save(Z_fused, fused_path)
                print(f"Fused embeddings saved to: {fused_path}")

                Embedding = Z_fused

            except Exception as e:
                print(f"ERROR during knowledge fusion: {e}")
                print("Falling back to original embedding...")
                import traceback
                traceback.print_exc()

    PPI = edge_list_data["PPI_edge_list"]
    PPI_dict = convert_ppi(PPI)
    PC_path = os.path.join(args.data_path, args.species, args.PC_path)
    PC = load_txt_list(PC_path, '/AdaPPI_golden_standard.txt')
    protein_dict = dict(zip(Protein_dict['Gene_symbol'], list(Protein_dict['ID'])))

    export_dir = args.export_dir if args.export_dir else os.path.join(args.output_dir, 'preds')
    variant = args.variant if args.variant else args.model

    results = HGC_DNN(
        PC, protein_dict, PPI_dict, Embedding,
        n_repeats=args.n_repeats,
        export_preds=args.export_preds,
        export_dir=export_dir,
        export_val=args.export_val,
        dataset=args.ppi_dataset,
        variant=variant,
        neg_ratio=args.neg_ratio,
        pooling_type=args.pooling_type
    )

    print("\n" + "="*60)
    print(f"Final Results for {args.ppi_dataset} PPI dataset:")
    print("="*60)
    print(results['message'])
    print(f"\nBinary Metrics:")
    for key, val in results['binary_metrics'].items():
        print(f"  {key}: {val:.4f}")
    print(f"\nComplex-level Metrics:")
    for key, val in results['complex_metrics'].items():
        print(f"  {key}: {val:.4f}")

    # Save results to JSON file
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f"results_{args.ppi_dataset}_{args.model}.json")

    results['config'] = {
        'ppi_dataset': args.ppi_dataset,
        'model': args.model,
        'epochs': args.epochs,
        'lr': args.lr,
        'hidden1': args.hidden1,
        'hidden2': args.hidden2,
        'droprate': args.droprate,
        'timestamp': str(datetime.now())
    }

    if args.model == 'StructStats':
        results['config']['no_spectral'] = True
        results['config']['she_k'] = args.she_k if args.she_k is not None else 64
        results['config']['aligned_by'] = 'PCA_or_ZeroPad'
        results['config']['she_r'] = None
        results['config']['she_T'] = None
        results['config']['she_alpha'] = None
        results['config']['note'] = 'StructStats: non-spectral structural features only'
    elif args.model == 'SHE':
        results['config']['she_r'] = args.she_r if args.she_r is not None else 64
        results['config']['she_k'] = args.she_k if args.she_k is not None else 64
        results['config']['she_T'] = args.she_T
        results['config']['she_alpha'] = args.she_alpha
        results['config']['vol_type'] = args.vol_type

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n Results saved to: {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Global parameters
    parser.add_argument('--species', type=str, default="Saccharomyces_cerevisiae", help="which species to use.")
    parser.add_argument('--data_path', type=str, default="./data", help="path storing data.")
    parser.add_argument('--feature_path', type=str, default="protein_feature", help="feature path data")
    parser.add_argument('--PPI_path', type=str, default="PPI", help="PPI data path")
    parser.add_argument('--PC_path', type=str, default="protein_complex", help="Protein complex data path")
    parser.add_argument('--model', type=str, default="SHE",
                        choices=['Node2vec', 'SHE', 'StructStats', 'None'],
                        help="Feature coding")

    # PPI dataset selection
    parser.add_argument('--ppi_dataset', type=str, default="Mann",
                        choices=['DIP', 'BioGRID', 'Mann'],
                        help="PPI dataset to use: DIP, BioGRID, or Mann")

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--hidden1', type=int, default=200, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=100, help="Number of units in hidden layer 2.")
    parser.add_argument('--droprate', type=float, default=0.5, help="Dropout rate (1 - keep probability).")
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs recorded in the result config.")

    # Output options
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save results")
    parser.add_argument('--save_model', action='store_true', help="Save trained model")

    # SHE parameters
    parser.add_argument('--she_r', type=int, default=None, help="SHE SVD rank")
    parser.add_argument('--she_k', type=int, default=None, help="SHE embedding dimension")
    parser.add_argument('--she_T', type=int, default=10, help="SHE window size")
    parser.add_argument('--she_alpha', type=float, default=0.1, help="SHE alpha parameter")
    parser.add_argument('--vol_type', type=str, default='traditional',
                        choices=['traditional', 'normalized'],
                        help="Volume type for structural basis scaling: "
                             "'traditional' uses sum of vertex degrees (H.sum()); "
                             "'normalized' uses sum of D_e^{-1/2}HD_v^{-1/2}")

    # Knowledge fusion parameters
    parser.add_argument('--use_keft_fusion', action='store_true', default=True , help="Enable knowledge-enhanced feature fusion")
    parser.add_argument('--fusion_method', type=str, default='ae', choices=['ae', 'concat'],
                        help="Fusion method")
    parser.add_argument('--fusion_dim', type=int, default=128,
                        help="Dimension of fused embeddings")
    parser.add_argument('--fusion_epochs', type=int, default=100,
                        help="Training epochs for autoencoder fusion")
    parser.add_argument('--fusion_lr', type=float, default=1e-3,
                        help="Learning rate for autoencoder fusion")
    parser.add_argument('--keft_T', type=int, default=12,
                        help="Number of temporal windows for activity features")
    parser.add_argument('--keft_k', type=int, default=1,
                        help="Threshold coefficient for temporal activity detection")
    parser.add_argument('--keft_cache', action='store_true', default=True,
                        help="Enable caching for knowledge features")
    parser.add_argument('--no_keft_cache', dest='keft_cache', action='store_false',
                        help="Disable caching (force rebuild)")

    # Reproducibility parameters
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

    # Evaluation parameters
    parser.add_argument('--n_repeats', type=int, default=30, help="Number of repeats with different negative sampling")
    parser.add_argument('--neg_ratio', type=int, default=5, help="Negative-to-positive ratio for sampling")
    parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'max', 'attention'],
                        help="Aggregation strategy for PCpredict: mean, max, or attention pooling")

    # Prediction export parameters
    parser.add_argument('--export_preds', action='store_true', default=True, help="Export predictions to CSV")
    parser.add_argument('--export_dir', type=str, default=None, help="Directory for exported predictions (default: output_dir/preds)")
    parser.add_argument('--export_val', action='store_true', default=True, help="Also export validation set predictions")
    parser.add_argument('--variant', type=str, default=None, help="Variant name for exported files (e.g., SHE_full, SHE_w_o_spectral)")

    args = parser.parse_args()

    # Set output directory based on dataset if using default
    if args.output_dir == "./results":
        args.output_dir = os.path.join("./results", args.ppi_dataset)

    print(args)
    print(f"\n{'='*60}")
    print(f"Training with PPI dataset: {args.ppi_dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    main(args)



