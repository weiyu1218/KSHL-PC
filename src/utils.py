from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import sys
import pickle as pkl
import scipy.sparse as sp
from evaluation import calculate_fmax
import networkx as nx
import os
import random
import pandas as pd
import numpy as np
import torch
import pickle
from pandas.core.frame import DataFrame

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1 :
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')

def find_amino_acid(x):
    """Filter sequences with ambiguous amino acids"""
    return ('B' in x) | ('O' in x) | ('J' in x) | ('U' in x) | ('X' in x) | ('Z' in x)

def CT(sequence):
    """Encode amino acid sequence using CT (Conjoint Triad) method"""
    classMap = {'G':'1','A':'1','V':'1','L':'2','I':'2','F':'2','P':'2',
            'Y':'3','M':'3','T':'3','S':'3','H':'4','N':'4','Q':'4','W':'4',
            'R':'5','K':'5','D':'6','E':'6','C':'7'}

    seq = ''.join([classMap[x] for x in sequence])
    length = len(seq)
    coding = np.zeros(343,dtype=np.int32)
    for i in range(length-2):
        index = int(seq[i]) + (int(seq[i+1])-1)*7 + (int(seq[i+2])-1)*49 - 1
        coding[index] = coding[index] + 1
    return coding

def sequence_CT(gene_entry_seq):
    """Apply CT encoding to protein sequences"""
    ambiguous_index = gene_entry_seq.loc[gene_entry_seq['Sequence'].apply(find_amino_acid)].index
    gene_entry_seq.drop(ambiguous_index, axis=0, inplace=True)
    gene_entry_seq.index = range(len(gene_entry_seq))
    print("after filtering:", gene_entry_seq.shape)
    print("encode amino acid sequence using CT...")
    CT_list = []
    for seq in gene_entry_seq['Sequence'].values:
        CT_list.append(CT(seq))
    gene_entry_seq['features_seq'] = CT_list

    return gene_entry_seq

def exact_gene_match(gene_name, sequence_df):
    """Match gene name to protein entry using exact matching"""
    gene_name_norm = str(gene_name).strip().upper()
    for idx, row in sequence_df.iterrows():
        gene_names_field = str(row['Gene Names'])
        if pd.isna(gene_names_field):
            continue
        gene_names_list = gene_names_field.replace(';', ' ').replace(',', ' ').split()
        gene_names_normalized = [g.strip().upper() for g in gene_names_list]
        if gene_name_norm in gene_names_normalized:
            return row['Entry']
    return 'NA'

def preprocessing_PPI(PPI,Sequence):
    """Filter and preprocess PPI network"""
    PPI_protein_list = list(set(PPI['protein1'].unique()).union(set(PPI['protein2'].unique())))
    PPI_protein = DataFrame(PPI_protein_list)
    PPI_protein['Entry'] = PPI_protein[0].apply(lambda x: exact_gene_match(x, Sequence))
    PPI_protein.columns = ['Gene_symbol', 'Entry']
    PPI_protein = PPI_protein[PPI_protein['Entry'] != 'NA']
    PPI_protein = PPI_protein[PPI_protein['Entry'].isin(set(Sequence['Entry']))]
    PPI_protein_list = list(set(PPI_protein['Gene_symbol'].unique()))
    PPI = PPI[PPI['protein1'].isin(PPI_protein_list)]
    PPI = PPI[PPI['protein2'].isin(PPI_protein_list)]
    PPI_protein_list = list(set(PPI['protein1'].unique()).union(set(PPI['protein2'].unique())))
    PPI_protein = PPI_protein[PPI_protein['Gene_symbol'].isin(PPI_protein_list)]
    PPI_protein = PPI_protein.sort_values(by=['Gene_symbol'])
    PPI_protein_list = list(PPI_protein['Gene_symbol'].unique())
    protein_dict = dict(zip(PPI_protein_list, list(range(0, len(PPI_protein_list)))))
    PPI['protein1'] = PPI['protein1'].apply(lambda x: protein_dict[x])
    PPI['protein2'] = PPI['protein2'].apply(lambda x: protein_dict[x])
    PPI_protein['ID'] = PPI_protein['Gene_symbol'].apply(lambda x: protein_dict[x])
    PPI_protein = PPI_protein.sort_values(by=['ID'])

    return PPI,PPI_protein

def is_sublist(sub_list, main_list):
    """Check if a list exists in a nested list"""
    for item in main_list:
        if isinstance(item, list) and sub_list == item:
            return True
    return False

def Nested_list_dup(Nested_list):
    """Remove duplicates from nested list"""
    Nested_list = [sorted(sublist) for sublist in Nested_list]
    Nested_list_dup = []
    for sublist in Nested_list:
        if sublist not in Nested_list_dup:
            Nested_list_dup.append(sublist)
    return Nested_list_dup

def count_sublist_lengths(lst):
    """Count distribution of nested list lengths"""
    length_counts = {}
    for sub_lst in lst:
        length = len(sub_lst)
        if length not in length_counts:
            length_counts[length] = 1
        else:
            length_counts[length] += 1
    return length_counts

def count_unique_elements(lst):
    """Count unique elements in nested list"""
    unique_elements = set()
    for sub_lst in lst:
        unique_elements |= set(sub_lst)
    return unique_elements

def convert_ppi(ppi_list):
    """Convert PPI list to dictionary format"""
    ppi_dict = {}
    for ppi in ppi_list:
        for protein in ppi:
            if protein not in ppi_dict:
                ppi_dict[protein] = []
            other_proteins = [p for p in ppi if p != protein]
            ppi_dict[protein].extend(other_proteins)
    return ppi_dict

def select_subunits(df, protein_list, fold, positive_set):
    """Sample negative complexes avoiding positive set"""
    complex_list = []
    for index, row in df.iterrows():
        subunit_count = row['subunits_counts']
        complex_count = row['counts_subunits_counts']
        attempts = 0
        max_attempts = complex_count * fold * 10
        for _ in range(complex_count * fold):
            while attempts < max_attempts:
                attempts += 1
                subunits = sorted(random.sample(protein_list, subunit_count))
                subunits_tuple = tuple(subunits)
                if subunits_tuple not in positive_set:
                    complex_list.append(subunits)
                    break
            if attempts >= max_attempts:
                subunits = sorted(random.sample(protein_list, subunit_count))
                complex_list.append(subunits)
    return complex_list

def negative_on_distribution(PC, protein_list, fold):
    """Generate negative samples following positive set size distribution"""
    PC_subunit_count = {}
    for pc in PC:
        length = len(pc)
        if length in PC_subunit_count:
            PC_subunit_count[length] += 1
        else:
            PC_subunit_count[length] = 1

    PC_subunit_count = pd.DataFrame(PC_subunit_count, index=[0]).T.reset_index()
    PC_subunit_count.columns = ['subunits_counts', 'counts_subunits_counts']

    # Create set of positive complexes for efficient lookup
    positive_set = set(tuple(sorted(pc)) for pc in PC)

    PCs_N = select_subunits(PC_subunit_count, protein_list, fold, positive_set)
    PCs_N = [sorted(i) for i in PCs_N]

    return PCs_N

def list_intersection(l1, l2):
    return list(set(l1).intersection(set(l2)))

def list_difference(l1, l2):
    return list(set(l1).difference(set(l2)))

def list_union(l1, l2):
    return list(set(l1).union(set(l2)))

def load_pickle(path, file_name):
    with open(f'{path}{file_name}', 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(path, file_name, data_pd):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(f'{path}{file_name}', 'wb') as f:
        pickle.dump(data_pd, f)

def load_txt_list(path, file_name, display_flag=True):
    if display_flag:
        print(f'Loading {path}{file_name}')

    list = []
    with open(f'{path}{file_name}', 'r') as f:

        lines = f.readlines()

        for line in lines:
            node_list = line.strip('\n').strip(' ').split(' ')
            list.append(node_list)

    return list

def calculate_fmax(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    a_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        p0 = (preds < threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        tn = np.sum(p0) - fn
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        acc = (tp + tn) / (tp + fp + tn + fn)
        if p_max < precision:
            f_max = f
            a_max = acc
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max, a_max, p_max, r_max, t_max

def calculate_f1_score(preds, labels):
    preds = preds.round()
    preds = preds.ravel()
    labels = labels.astype(np.int32)
    labels = labels.ravel()
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f = f1_score(labels, preds)
    return f, acc, precision, recall

def evaluate_performance(y_test, y_score):
    n_classes = y_test.shape[1]
    perf = dict()

    perf["M-aupr"] = 0.0
    n = 0
    aupr_list = []
    num_pos_list = []
    for i in range(n_classes):
        num_pos = sum(y_test[:, i])
        num_pos = num_pos.astype(float)
        if num_pos > 0:
            ap = average_precision_score(y_test[:, i], y_score[:, i])
            n += 1
            perf["M-aupr"] += ap
            aupr_list.append(ap)
            num_pos_list.append(num_pos)
    perf["M-aupr"] /= n
    perf['aupr_list'] = aupr_list
    perf['num_pos_list'] = num_pos_list

    # Compute micro-averaged AUPR
    perf['m-aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())
    perf['roc_auc'] = roc_auc_score(y_test.ravel(), y_score.ravel())
    perf['F-max'], perf['acc_max'], perf['pre_max_max'], perf['rec_max'], perf['thr_max'] = calculate_fmax(y_score, y_test)  #precision_max
    perf['F1-score'], perf['accuracy'], perf['precision'], perf['recall'] = calculate_f1_score(y_score, y_test)
    return perf

def plot_roc(Y_label, y_pred,str):
    fpr, tpr, threshold = roc_curve(Y_label, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, color='red',lw=2, label='sequence (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(str+' ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def read_ppi_file(file_path):
    with open(file_path, "r") as file:
        ppi_pairs = [line.strip().split() for line in file.readlines()]
    return ppi_pairs

def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))

def SparseTensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_ppi_network(filename, gene_num):
    with open(filename) as f:
        data = f.readlines()
    adj = np.zeros((gene_num, gene_num))
    for x in data:
        temp = x.strip().split("\t")
        adj[int(temp[0]), int(temp[1])] = 1
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T
    return adj

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    adj_sparse = sp.coo_matrix(adj_normalized)
    return adj_sparse

def mask_test_edges(adj):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def create_diagonal_1_array(arr):
        new_array = np.copy(arr)
        np.fill_diagonal(new_array, 1)
        return new_array

# PPI Extend
def _score_with_cache(model, X, subgraphs_list, score_cache=None, batch_size: int = 512):
    """Score a list of subgraphs using optional cache and batching.
    Returns list of floats in the same order as input.
    """
    if score_cache is None:
        score_cache = {}

    keys = [tuple(sorted(sg)) for sg in subgraphs_list]
    to_compute_idx = [i for i, k in enumerate(keys) if k not in score_cache]

    # compute missing items in batches
    if to_compute_idx:
        with torch.no_grad():
            for i in range(0, len(to_compute_idx), batch_size):
                idx_batch = to_compute_idx[i:i+batch_size]
                batch_subgraphs = [subgraphs_list[j] for j in idx_batch]
                probs = model(X, batch_subgraphs)
                probs = probs.detach().cpu().numpy().reshape(-1).tolist()
                for j, p in zip(idx_batch, probs):
                    score_cache[keys[j]] = float(p)

    return [score_cache[k] for k in keys]


def subgraph_expansion(subgraph, threshold_alpha, PPI_dict, model, X,
                       eps: float = 1e-6, max_size=None,
                       score_cache=None, batch_size: int = 512):
    expanded_subgraph = subgraph.copy()
    adjacent_points = [PPI_dict[i] for i in expanded_subgraph]
    adjacent_points = list(set([item for sublist in adjacent_points for item in sublist]))

    # initial score of current subgraph
    current_score = _score_with_cache(model, X, [expanded_subgraph], score_cache, batch_size)[0]

    while True:
        if max_size is not None and len(expanded_subgraph) >= max_size:
            break

        adjacent = list(set(adjacent_points).difference(set(expanded_subgraph)))
        if len(adjacent) == 0:
            break

        candidates = [expanded_subgraph + [v] for v in adjacent]
        scores = _score_with_cache(model, X, candidates, score_cache, batch_size)
        max_index = int(np.argmax(scores))
        best_new_score = scores[max_index]

        # Require both: above alpha and strictly improves over current
        if best_new_score > max(threshold_alpha, current_score + eps):
            expanded_subgraph.append(adjacent[max_index])
            current_score = best_new_score
        else:
            break

    return expanded_subgraph

def nested_list_unique(nested_list):
    nested_list = [sorted(sublist) for sublist in nested_list]
    nested_list_dup = []
    for sublist in nested_list:
        if sublist not in nested_list_dup:
            nested_list_dup.append(sublist)
    return nested_list_dup

def calculate_overlap_ratio(subgraph_i, subgraph_k):
    intersection = len(set(subgraph_i).intersection(set(subgraph_k)))
    union = len(set(subgraph_i).union(set(subgraph_k)))
    overlap_ratio = intersection / union
    return overlap_ratio

def subgraph_filtration(candidate_subgraphs, scores, threshold_beta, model, X,
                        score_cache=None, batch_size: int = 512):
    sorted_complexes = sorted(zip(candidate_subgraphs, scores), key=lambda x: x[1], reverse=True)
    filtered_PC = []

    for i in range(len(sorted_complexes)):
        candidate_subgraph_i, score_i = sorted_complexes[i]
        if len(candidate_subgraph_i) != 0:
            to_score = []
            idx_map = []
            filtered_PC_i = []
            for k in range(i+1,len(sorted_complexes)):
                candidate_subgraph_k, score_j = sorted_complexes[k]
                overlapping_ratio = calculate_overlap_ratio(candidate_subgraph_i, candidate_subgraph_k)
                if overlapping_ratio >= threshold_beta:
                    candidate_subgraph = list(set(candidate_subgraph_i).union(set(candidate_subgraph_k)))
                    to_score.append(candidate_subgraph)
                    idx_map.append(k)

            if to_score:
                merged_scores = _score_with_cache(model, X, to_score, score_cache, batch_size)
                for candidate_subgraph, k, merged_score in zip(to_score, idx_map, merged_scores):
                    if merged_score > score_i:
                        filtered_PC_i.append(candidate_subgraph)
                    else:
                        sorted_complexes[k] = [],[]

            if len(filtered_PC_i) == 0:
                filtered_PC.append(candidate_subgraph_i)
            else:
                filtered_PC.extend(filtered_PC_i)

    return filtered_PC



