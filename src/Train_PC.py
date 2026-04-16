from utils import *
from sklearn.model_selection import KFold
from evaluation import calculate_fmax,get_score
from models import PCpredict
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve,roc_auc_score
from sklearn.metrics import auc
import json
import os
import csv
from datetime import datetime

def train_DNN(X, Y_train, Train_PC, Test_PC, epochs, learning_rate, drop_rate, weight_decay=1e-4, pooling_type='mean'):
    model = PCpredict(int(X.shape[1]), 1, drop_rate=drop_rate, pooling_type=pooling_type).to(device=try_gpu())
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    Y_train = Y_train.to(device=try_gpu())

    for e in range(epochs):
        model.train()
        out = model(X, Train_PC)
        loss_train = loss(out, Y_train)
        optimizer.zero_grad()
        loss_train.backward(retain_graph=True)
        optimizer.step()

        if e % 100 == 0:
            print(f"Epoch: {e + 1}; train_loss: {loss_train.data:.4f};")
    model.eval()
    out_valid = model(X, Test_PC)
    return out_valid, model

def HGC_DNN(PC, protein_dict, PPI_dict, X, n_repeats=30, export_preds=False, export_dir='./preds', export_val=False, dataset='Mann', variant='SHE', neg_ratio=5, pooling_type='mean'):
    # Convert complex IDs to protein indices
    PCs = []
    for pc in PC:
        if len(pc) > 2:
            if set(pc).issubset(set(list(protein_dict.keys()))):
                pc_map = [protein_dict[sub] for sub in pc]
                PCs.append(pc_map)
    PC = [sorted(i) for i in PCs]

    # Setup export paths
    if export_preds:
        os.makedirs(export_dir, exist_ok=True)

        preds_csv = os.path.join(export_dir, f'{variant}_predictions.csv')
        complexes_csv = os.path.join(export_dir, f'{variant}_complexes.csv')

        preds_file = open(preds_csv, 'w', newline='')
        preds_writer = csv.writer(preds_file)
        preds_writer.writerow(['dataset', 'model', 'variant', 'repeat', 'fold', 'split', 'y_true', 'y_score', 'threshold'])

        complexes_file = open(complexes_csv, 'w', newline='')
        complexes_writer = csv.writer(complexes_file)
        complexes_writer.writerow(['dataset', 'variant', 'repeat', 'fold', 'complex_members', 'score'])

    # Print data statistics
    print(f"\n{'='*60}")
    print("Dataset Statistics:")
    print(f"{'='*60}")
    print(f"Total proteins in PPI network: {len(PPI_dict)}")
    print(f"Total PPI edges: {sum(len(v) for v in PPI_dict.values()) // 2}")
    print(f"Positive complexes (filtered, size>=3): {len(PC)}")

    # Complex size distribution
    from collections import Counter
    pc_sizes = Counter(len(pc) for pc in PC)
    print(f"Complex size distribution: {dict(sorted(pc_sizes.items()))}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"{'='*60}\n")

    # Store metrics across all repeats
    all_repeats_metrics = {
        'f1': [], 'precision': [], 'recall': [], 'sensitivity': [],
        'specificity': [], 'accuracy': [], 'threshold': [],
        'auprc': [], 'auroc': [],
        'complex_precision': [], 'complex_recall': [], 'complex_f1': [],
        'complex_acc': [], 'complex_sn': [], 'complex_ppv': []
    }

    print(f"\nRunning {n_repeats} repeats with different negative sampling...")

    for repeat_idx in range(n_repeats):
        print(f"\n{'='*60}")
        print(f"Repeat {repeat_idx + 1}/{n_repeats}")
        print(f"{'='*60}")

        label1 = torch.ones(len(PC), 1, dtype=torch.float)

        # Store fold-level metrics for this repeat
        fold_metrics = {
            'f1': [], 'precision': [], 'recall': [], 'sensitivity': [],
            'specificity': [], 'accuracy': [], 'threshold': [],
            'auprc': [], 'auroc': []
        }

        # Five-Fold cross validation with fixed random_state for reproducibility
        all_idx = list(range(len(label1)))
        rs = KFold(n_splits=5, shuffle=True, random_state=42 + repeat_idx)
        cv_index_set = rs.split(all_idx)

        fold_num = 0
        all_predict_pc_per_repeat = []

        for train_index, test_index in cv_index_set:
            fold_num += 1
            print(f'This is the {fold_num} Fold')
            train_index = train_index.tolist()
            test_index = test_index.tolist()

            # Split train_index into train and validation (8:2)
            np.random.seed(42 + repeat_idx + fold_num)
            np.random.shuffle(train_index)
            val_size = int(len(train_index) * 0.2)
            val_index = train_index[:val_size]
            train_only_index = train_index[val_size:]

            # Training set (80% of original train)
            Train_PC = [PC[i] for i in train_only_index]
            Train_label1 = torch.ones(len(Train_PC), 1, dtype=torch.float)
            Train_PC_negative = negative_on_distribution(Train_PC, list(PPI_dict.keys()), neg_ratio)
            Train_label0 = torch.zeros(len(Train_PC_negative), 1, dtype=torch.float)
            Train_labels = torch.cat((Train_label1, Train_label0), dim=0)
            Train_PC_PN = Train_PC + Train_PC_negative
            all_idx_train = list(range(len(Train_PC_PN)))
            np.random.shuffle(all_idx_train)
            Train_PC_PN = [Train_PC_PN[i] for i in all_idx_train]
            Train_labels = Train_labels[all_idx_train]
            print(f'    Train: {len(Train_PC)} pos + {len(Train_PC_negative)} neg = {len(Train_PC_PN)} total (ratio 1:{len(Train_PC_negative)/len(Train_PC):.1f})')

            # Validation set (20% of original train)
            Val_PC = [PC[i] for i in val_index]
            Val_label1 = torch.ones(len(Val_PC), 1, dtype=torch.float)
            Val_PC_negative = negative_on_distribution(Val_PC, list(PPI_dict.keys()), neg_ratio)
            Val_label0 = torch.zeros(len(Val_PC_negative), 1, dtype=torch.float)
            Val_labels = torch.cat((Val_label1, Val_label0), dim=0)
            Val_PC_PN = Val_PC + Val_PC_negative
            all_idx_val = list(range(len(Val_PC_PN)))
            np.random.shuffle(all_idx_val)
            Val_PC_PN = [Val_PC_PN[i] for i in all_idx_val]
            Val_labels = Val_labels[all_idx_val]
            print(f'    Val: {len(Val_PC)} pos + {len(Val_PC_negative)} neg = {len(Val_PC_PN)} total (ratio 1:{len(Val_PC_negative)/len(Val_PC):.1f})')

            # Test set
            Test_PC = [PC[i] for i in test_index]
            Test_label1 = torch.ones(len(Test_PC), 1, dtype=torch.float)
            Test_PC_negative = negative_on_distribution(Test_PC, list(PPI_dict.keys()), neg_ratio)
            Test_label0 = torch.zeros(len(Test_PC_negative), 1, dtype=torch.float)
            Test_labels = torch.cat((Test_label1, Test_label0), dim=0)
            Test_PC_PN = Test_PC + Test_PC_negative
            all_idx_test = list(range(len(Test_PC_PN)))
            np.random.shuffle(all_idx_test)
            Test_PC_PN = [Test_PC_PN[i] for i in all_idx_test]
            Test_labels = Test_labels[all_idx_test]
            print(f'    Test: {len(Test_PC)} pos + {len(Test_PC_negative)} neg = {len(Test_PC_PN)} total (ratio 1:{len(Test_PC_negative)/len(Test_PC):.1f})')

            # Train model
            y_pred_train, model = train_DNN(X, Train_labels, Train_PC_PN, Train_PC_PN, 500, 0.001, 0.4, weight_decay=5e-4, pooling_type=pooling_type)

            # Get validation predictions to select threshold for THIS fold
            model.eval()
            y_val = model(X, Val_PC_PN)
            y_val_np = y_val.data.cpu().numpy()
            val_labels_np = Val_labels.cpu().numpy()

            # Select threshold on validation set for this fold
            _, fold_threshold, _, _, _, _, _ = calculate_fmax(y_val_np, val_labels_np)
            print(f'  Fold {fold_num} threshold from validation: {fold_threshold:.4f}')

            # Get test predictions for this fold
            y_pred = model(X, Test_PC_PN)
            y_pred_np = y_pred.data.cpu().numpy()
            test_labels_np = Test_labels.cpu().numpy()

            # Calculate fold-level binary metrics using the fold's threshold
            predictions = (y_pred_np > fold_threshold).astype(np.int32)
            tp = np.sum(predictions * test_labels_np.astype(np.int32))
            fp = np.sum(predictions) - tp
            fn = np.sum(test_labels_np) - tp
            tn = np.sum((1 - predictions) * (1 - test_labels_np.astype(np.int32)))

            fold_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            fold_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fold_f1 = 2 * fold_precision * fold_recall / (fold_precision + fold_recall) if (fold_precision + fold_recall) > 0 else 0
            fold_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            fold_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fold_acc = (tp + tn) / (tp + fp + tn + fn)

            # Calculate fold-level AUPRC and AUROC
            from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
            precision_curve, recall_curve, _ = precision_recall_curve(test_labels_np, y_pred_np)
            fold_auprc = auc(recall_curve, precision_curve)
            fold_auroc = roc_auc_score(test_labels_np, y_pred_np)

            # Store fold metrics
            fold_metrics['f1'].append(fold_f1)
            fold_metrics['precision'].append(fold_precision)
            fold_metrics['recall'].append(fold_recall)
            fold_metrics['sensitivity'].append(fold_sensitivity)
            fold_metrics['specificity'].append(fold_specificity)
            fold_metrics['accuracy'].append(fold_acc)
            fold_metrics['threshold'].append(fold_threshold)
            fold_metrics['auprc'].append(fold_auprc)
            fold_metrics['auroc'].append(fold_auroc)

            # Collect predicted complexes for this fold
            fold_predict_pc = [Test_PC_PN[i] for i in range(len(y_pred_np)) if y_pred_np[i] > fold_threshold]
            all_predict_pc_per_repeat.extend(fold_predict_pc)

            # Export predictions if enabled
            if export_preds:
                # Export validation predictions if requested
                if export_val:
                    for i in range(len(y_val_np)):
                        preds_writer.writerow([
                            dataset, 'PCpredict', variant, repeat_idx + 1, fold_num,
                            'val', val_labels_np[i][0], y_val_np[i][0], fold_threshold
                        ])

                # Export test predictions
                for i in range(len(y_pred_np)):
                    preds_writer.writerow([
                        dataset, 'PCpredict', variant, repeat_idx + 1, fold_num,
                        'test', test_labels_np[i][0], y_pred_np[i][0], fold_threshold
                    ])

                # Export predicted complexes
                for pc in fold_predict_pc:
                    pc_idx = Test_PC_PN.index(pc)
                    score = y_pred_np[pc_idx][0]
                    complex_str = ';'.join(map(str, pc))
                    complexes_writer.writerow([
                        dataset, variant, repeat_idx + 1, fold_num, complex_str, score
                    ])


        # Calculate 5-fold average for this repeat
        repeat_f1 = np.mean(fold_metrics['f1'])
        repeat_precision = np.mean(fold_metrics['precision'])
        repeat_recall = np.mean(fold_metrics['recall'])
        repeat_sensitivity = np.mean(fold_metrics['sensitivity'])
        repeat_specificity = np.mean(fold_metrics['specificity'])
        repeat_acc = np.mean(fold_metrics['accuracy'])
        repeat_threshold = np.mean(fold_metrics['threshold'])
        repeat_auprc = np.mean(fold_metrics['auprc'])
        repeat_auroc = np.mean(fold_metrics['auroc'])

        # Log threshold distribution for this repeat
        threshold_min = np.min(fold_metrics['threshold'])
        threshold_max = np.max(fold_metrics['threshold'])
        threshold_std = np.std(fold_metrics['threshold'], ddof=1)

        print(f"\nRepeat {repeat_idx + 1} 5-fold average:")
        print(f"  F1: {repeat_f1:.4f}, AUPRC: {repeat_auprc:.4f}, AUROC: {repeat_auroc:.4f}")
        print(f"  Threshold: {repeat_threshold:.4f} (min={threshold_min:.4f}, max={threshold_max:.4f}, std={threshold_std:.4f})")

        # Calculate complex-level metrics for this repeat
        complex_precision, complex_recall, complex_f1, complex_acc, complex_sn, complex_PPV, score = get_score(PC, all_predict_pc_per_repeat)
        print(f"  Complex metrics: {score}")

        # Store metrics for this repeat
        all_repeats_metrics['f1'].append(repeat_f1)
        all_repeats_metrics['precision'].append(repeat_precision)
        all_repeats_metrics['recall'].append(repeat_recall)
        all_repeats_metrics['sensitivity'].append(repeat_sensitivity)
        all_repeats_metrics['specificity'].append(repeat_specificity)
        all_repeats_metrics['accuracy'].append(repeat_acc)
        all_repeats_metrics['threshold'].append(repeat_threshold)
        all_repeats_metrics['auprc'].append(repeat_auprc)
        all_repeats_metrics['auroc'].append(repeat_auroc)
        all_repeats_metrics['complex_precision'].append(complex_precision)
        all_repeats_metrics['complex_recall'].append(complex_recall)
        all_repeats_metrics['complex_f1'].append(complex_f1)
        all_repeats_metrics['complex_acc'].append(complex_acc)
        all_repeats_metrics['complex_sn'].append(complex_sn)
        all_repeats_metrics['complex_ppv'].append(complex_PPV)

    # Calculate mean and standard error for all metrics
    print(f"\n{'='*60}")
    print("Aggregated Results across all repeats:")
    print(f"{'='*60}")

    def calc_mean_se(values):
        mean = np.mean(values)
        if len(values) > 1:
            se = np.std(values, ddof=1) / np.sqrt(len(values))
        else:
            se = 0.0
        return mean, se

    results = {
        'binary_metrics': {},
        'complex_metrics': {},
        'message': f"Results aggregated from {n_repeats} repeats with different negative sampling"
    }

    # Binary metrics
    for metric in ['f1', 'precision', 'recall', 'sensitivity', 'specificity', 'accuracy', 'threshold', 'auprc', 'auroc']:
        mean, se = calc_mean_se(all_repeats_metrics[metric])
        results['binary_metrics'][metric] = float(mean)
        results['binary_metrics'][f'{metric}_se'] = float(se)
        print(f"Binary {metric}: {mean:.4f} 卤 {se:.4f}")

    # Complex-level metrics
    for metric in ['precision', 'recall', 'f1', 'acc', 'sn', 'ppv']:
        full_metric = f'complex_{metric}'
        mean, se = calc_mean_se(all_repeats_metrics[full_metric])
        results['complex_metrics'][metric] = float(mean)
        results['complex_metrics'][f'{metric}_se'] = float(se)
        print(f"Complex {metric}: {mean:.4f} 卤 {se:.4f}")

    # Close export files if opened
    if export_preds:
        preds_file.close()
        complexes_file.close()
        print(f"\nPredictions exported to: {preds_csv}")
        print(f"Complexes exported to: {complexes_csv}")

    return results



