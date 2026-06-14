import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix as sk_contingency
from dataclasses import dataclass
from typing import Dict, Any, List

from src.community import add_similarity_weight_from_length, louvain_partition

import warnings
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.*",
    category=UserWarning,
)

def cut_graph_by_length_threshold(
    G: nx.Graph,
    thr: float,
    length_attr: str = "length",
) -> nx.Graph:
    """
    Return a subgraph of G containing edges with length <= thr.
    - Inputs:
        - G: input graph
        - thr: length threshold
        - length_attr: edge attribute name for lengths
    - Returns:
        - H: subgraph with edges satisfying length <= thr
    - Procedure:
        - Create an empty graph H
        - Add all nodes from G to H
        - For each edge (u, v) in G:
            - If length_attr of edge <= thr, add edge (u, v) to H
        - Return H
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        L = float(d.get(length_attr, 1.0))
        if L <= thr:
            H.add_edge(u, v)
    return H

def components_as_partition(H: nx.Graph) -> dict:
    """
    Return dict node -> component_id based on connected components.
    """
    part = {}
    for cid, comp in enumerate(nx.connected_components(H)):
        for n in comp:
            part[n] = cid
    return part

def best_ari_by_length_cut_k(
    G_flow: nx.Graph,
    gt: dict,
    k_target: int = 2,
    length_attr: str = "length",
    thr_grid: np.ndarray | None = None,
    n_thr: int = 200,
    q_min: float = 0.0,
    q_max: float = 1.0,
    debug: bool = True,
    progress_every: int = 50,
    fallback: str = "closest",
):
    """
    Threshold-scan Community Cut:
      - For each thr: cut edges with length > thr (keep <= thr)
      - Partition = connected components
      - Compute ARI(partition, gt)
      - Prefer solutions with k_pred == k_target
    Return: (best_ari, best_thr, best_k, best_part)
    """
    # --- build threshold grid ---
    if thr_grid is None:
        lens = np.array([float(d.get(length_attr, 1.0)) for _, _, d in G_flow.edges(data=True)], dtype=float)
        if lens.size == 0:
            # no edges => everything isolated
            part = {n: i for i, n in enumerate(G_flow.nodes())}
            ari = ari_from_partition(part, gt)
            return ari, 0.0, len(part), part

        qs = np.linspace(float(q_min), float(q_max), int(n_thr))
        thr_grid = np.unique(np.quantile(lens, qs))
    else:
        thr_grid = np.asarray(thr_grid, dtype=float)

    records = []  # (thr, ari, k_pred, part)

    for i, thr in enumerate(thr_grid):
        if debug and progress_every > 0 and (i % int(progress_every) == 0):
            print(f"[cut-eval] {i}/{len(thr_grid)} thr={float(thr):.4f}", flush=True)

        H = cut_graph_by_length_threshold(G_flow, float(thr), length_attr=length_attr)
        part = components_as_partition(H)
        ari = ari_from_partition(part, gt)
        k_pred = len(set(part.values()))

        records.append((float(thr), float(ari), int(k_pred), part))

    strict = [r for r in records if r[2] == int(k_target)]

    if debug:
        print(f"[cut-debug] k_target={k_target} strict={len(strict)}/{len(records)}", flush=True)

    if len(strict) > 0:
        # choose max ARI among strict
        best_thr, best_ari, best_k, best_part = max(strict, key=lambda x: x[1])
        return best_ari, best_thr, best_k, best_part

    # --- fallback ---
    if fallback == "best_ari":
        best_thr, best_ari, best_k, best_part = max(records, key=lambda x: x[1])
        return best_ari, best_thr, best_k, best_part

    best_gap = 10**9
    best = None
    for thr, ari, k, part in records:
        gap = abs(int(k) - int(k_target))
        if (gap < best_gap) or (gap == best_gap and (best is None or ari > best[1])):
            best_gap = gap
            best = (thr, ari, k, part)

    best_thr, best_ari, best_k, best_part = best
    return best_ari, best_thr, best_k, best_part


def relabel_to_compact_int(labels: Dict[int, Any]) -> Dict[int, int]:
    """Map arbitrary labels to 0..K-1"""
    uniq = sorted(set(labels.values()))
    mp = {u: i for i, u in enumerate(uniq)}
    return {n: mp[v] for n, v in labels.items()}

def get_y_true_from_gt(gt: dict) -> np.ndarray:
    '''
    gt: ground-truth labels dict {node: label}
    Returns y_true array aligned by sorted node order.
    '''
    gt2 = relabel_to_compact_int(gt)
    nodes = sorted(gt2.keys())
    return np.array([gt2[n] for n in nodes], dtype=int)

def ari_from_partition(part, gt):
    """
    part: predicted partition (dict or list-of-communities)
    gt  : ground-truth labels dict {node: label}
    """
    # --- convert part to dict {node: community_id} ---
    if isinstance(part, dict):
        pred = part
    else:
        # assume list of communities (each is iterable of nodes)
        pred = {}
        for cid, comm in enumerate(part):
            for node in comm:
                pred[node] = cid

    nodes = sorted(set(gt.keys()) & set(pred.keys()))
    if len(nodes) == 0:
        return 0.0

    y_true = [gt[n] for n in nodes]
    y_pred = [pred[n] for n in nodes]

    return float(adjusted_rand_score(y_true, y_pred))


def calc_ari_with_labels(labels: list, gt:dict) -> float:
    '''
    Calculate Adjusted Rand Index (ARI) between predicted labels and ground truth.
    - Inputs:
        - labels: list of predicted labels aligned by sorted node order
        - gt: ground-truth labels dict {node: label}
    - Outputs:
        - ARI score as float
    '''
    y_true = get_y_true_from_gt(gt)
    return float(adjusted_rand_score(labels, y_true))


def complete_partition(part: dict, nodes: List[int]) -> dict:
    """
    Given a partition dict (node -> community_id) that may be incomplete,
    return a complete partition dict for the specified nodes.
    Missing nodes are assigned to new unique community IDs.
    - Inputs:
        - part: dict {node: community_id}
        - nodes: list of all nodes to cover
    - Returns:
        - part2: complete dict {node: community_id}
    - Procedure:
        - Copy the input partition to part2
        - Identify nodes in 'nodes' not present in part2
        - For each missing node, assign a new community ID starting from max existing ID + 1
        - Return part2
    """
    part2 = dict(part)  # copy
    missing = [n for n in nodes if n not in part2]
    if missing:
        base = (max(part2.values()) + 1) if len(part2) > 0 else 0
        for i, n in enumerate(missing):
            part2[n] = base + i
    return part2

def best_ari_by_louvain_resolution_k(
    G_flow: nx.Graph,
    gt: dict,
    k_target: int = 2,
    length_attr: str = "length",
    beta: float = 1.0,
    weight_attr: str = "weight",
    res_grid: np.ndarray | None = None,
    n_res: int = 60,
    res_min: float = 0.2,
    res_max: float = 4.0,
    seed: int = 0,
    debug: bool = True,
    progress_every: int = 20,
):
    '''
    Search for the best Louvain resolution parameter to maximize ARI with respect to gt,
    while keeping the predicted number of communities equal to k_target if possible.
    Returns (best_ari, best_resolution, k_predicted, best_partition).

    - Inputs:
        - G_flow: input graph for Louvain
        - gt: ground-truth labels dict {node: label}
        - k_target: target number of communities
        - length_attr: edge attribute name for lengths
        - beta: parameter for similarity weight computation
        - weight_attr: edge attribute name for weights
        - res_grid: optional array of resolution parameters to try
        - n_res: number of resolution parameters to generate if res_grid is None
        - res_min: minimum resolution value if res_grid is None
        - res_max: maximum resolution value if res_grid is None
        - seed: random seed for Louvain
        - debug: whether to print debug information
        - progress_every: frequency of progress logging
    - Outputs:
        - best_ari: best Adjusted Rand Index score
        - best_resolution: resolution parameter that achieved best ARI
        - k_predicted: number of communities in the best partition
        - best_partition: partition dict {node: community_id} for best ARI
    - Procedure:
        - Prepare graph by adding similarity weights based on lengths
        - Generate resolution grid if not provided
        - For each resolution in the grid:
            - Compute Louvain partition
            - Calculate ARI with respect to gt
            - Record resolution, ARI, and number of predicted communities
        - Select the best partition that matches k_target if possible
        - If no exact match, select the partition closest to k_target with highest ARI
        - Return best ARI, resolution, predicted community count, and partition
    '''
    G = G_flow.copy()
    add_similarity_weight_from_length(G, length_attr=length_attr, weight_attr=weight_attr, beta=beta)

    debug_weight_stats(
        G,
        length_attr=length_attr,
        weight_attr=weight_attr,
        prefix=f"[louvain][beta={beta:.2g}] "
    )

    if res_grid is None:
        res_grid = np.linspace(res_min, res_max, int(n_res), dtype=float)
    else:
        res_grid = np.asarray(res_grid, dtype=float)

    records = []
    strict = []

    for i, res in enumerate(res_grid):
        if debug and progress_every > 0 and (i % int(progress_every) == 0):
            print(f"[louvain-eval] {i}/{len(res_grid)} res={float(res):.3f}", flush=True)

        part = louvain_partition(G, weight_attr=weight_attr, resolution=float(res), seed=int(seed))
        ari = ari_from_partition(part, gt)
        k_pred = len(set(part.values()))

        print(f"[louvain-grid] i={i:02d} res={float(res):.3f} k={k_pred} ari={ari:.4f} strict={k_pred==int(k_target)}",
                flush=True)

        records.append((float(res), float(ari), int(k_pred)))
        if k_pred == int(k_target):
            strict.append((float(res), float(ari), int(k_pred)))

    if debug:
        print(f"[louvain-debug] k_target={k_target} seen={len(strict)}/{len(records)}", flush=True)

    if len(strict) > 0:
        best = max(strict, key=lambda x: x[1])
        best_part = louvain_partition(G, weight_attr=weight_attr, resolution=float(best[0]), seed=int(seed))
        return best[1], best[0], best[2], best_part  ## best resolution, best ARI, cluster number and best partition

    best = min(records, key=lambda x: (abs(x[2] - int(k_target)), -x[1]))

    best_part = louvain_partition(G, weight_attr, resolution=best[0], seed=seed)
    return best[1], best[0], best[2], best_part ## best resolution, best ARI, cluster number and best partition

def contingency_matrix(gt: dict, pred: dict) -> None:
    '''
    Compute and print a 2x2 contingency matrix for binary ground truth and predicted partitions.
    - Inputs:
        - gt: ground-truth labels dict {node: label} (binary: 0/1)
        - pred: predicted labels dict {node: label} (binary: 0/1)
    - Outputs:
        - Prints the contingency matrix in the format:
          [[c00, c01],
           [c10, c11]]
          where cij represents counts of (true_label=i, pred_label=j)
    - Procedure:
        - Relabel gt to compact integers
        - Align nodes in sorted order
        - Create binary arrays y_true and y_pred
        - Compute counts c00, c01, c10, c11
        - Print the contingency matrix
    '''
    gt2 = relabel_to_compact_int(gt)
    nodes = sorted(gt2.keys())

    y_true = np.array([gt2[n] for n in nodes], dtype=int)   # 0/1
    y_pred = np.array([pred[n] for n in nodes], dtype=int)  # 0/1 (k=2)

    # 2x2 contingency
    c00 = int(np.sum((y_true==0) & (y_pred==0)))
    c01 = int(np.sum((y_true==0) & (y_pred==1)))
    c10 = int(np.sum((y_true==1) & (y_pred==0)))
    c11 = int(np.sum((y_true==1) & (y_pred==1)))
    print(f"[cont] [[{c00},{c01}],[{c10},{c11}]]", flush=True)

def contingency_matrix_general(gt: dict, pred: dict, top_k: int = 10):
    '''
    Compute and return the contingency matrix between ground truth and predicted partitions.
    - Inputs:
        - gt: ground-truth labels dict {node: label}
        - pred: predicted labels dict {node: label}
        - top_k: int, number of top rows/columns to keep in the output
    - Outputs:
        - df: pandas DataFrame representing the contingency matrix
    - Procedure:
        - Align nodes in sorted order
        - Create arrays y_true and y_pred
        - Compute contingency matrix C
        - Convert C to pandas DataFrame
        - Print shape information
        - Compute and sort row and column sums
        - Return the DataFrame
    '''
    nodes = sorted(gt.keys())
    y_true = np.array([gt[n] for n in nodes], dtype=int)
    pred2 = complete_partition(pred, nodes)
    y_pred = np.array([pred2[n] for n in nodes], dtype=int)

    C = sk_contingency(y_true, y_pred)

    df = pd.DataFrame(C)
    print(f"[contingency] shape={df.shape}  (true={df.shape[0]} pred={df.shape[1]})")

    row_sum = df.sum(axis=1).sort_values(ascending=False)
    col_sum = df.sum(axis=0).sort_values(ascending=False)

    return df

@dataclass
class WeightStats:
    beta: float
    n_edges: int
    w_min: float
    w_p01: float
    w_p05: float
    w_med: float
    w_mean: float
    w_p95: float
    w_p99: float
    w_max: float
    n_eps: int

def _edge_attr_array(G: nx.Graph, attr: str) -> np.ndarray:
    '''
    Extract an array of edge attribute values from the graph.
    - Inputs:
        - G: input graph
        - attr: edge attribute name to extract
    - Outputs:
        - arr: numpy array of attribute values (float)
    - Procedure:
        - Initialize an empty list arr
        - For each edge in G:
            - Get the attribute value
            - If the value is not None, convert to float and append to arr
        - Convert arr to a numpy array and return
    '''
    arr = []
    for _, _, d in G.edges(data=True):
        v = d.get(attr, None)
        if v is None:
            continue
        try:
            arr.append(float(v))
        except Exception:
            pass
    return np.asarray(arr, dtype=float)

def mst_from_length_with_label_shuffle(
    G: nx.Graph,
    length_attr: str = "length",
    seed: int = 0,
) -> nx.Graph:
    """
    Create a Minimum Spanning Tree (MST) from graph G by shuffling node labels.
    - Inputs:
        - G: input graph
        - length_attr: edge attribute name for lengths
        - seed: random seed for shuffling
    - Outputs:
        - T: MST graph with original node labels
    - Procedure:
        - Extract list of nodes from G
        - Initialize random number generator with seed
        - Create a shuffled permutation of nodes
        - Create a mapping from original nodes to shuffled nodes
        - Create inverse mapping from shuffled nodes back to original nodes
        - Relabel nodes of G using the mapping to create G2
        - Compute the MST of G2 using the specified length attribute
        - Relabel nodes of the MST back to original labels using the inverse mapping to create T
        - Ensure T contains all original nodes (in case of isolated nodes)
        - Return T
    """
    nodes = list(G.nodes())
    rng = np.random.default_rng(int(seed))
    perm = nodes.copy()
    rng.shuffle(perm)

    mapping = {old: new for old, new in zip(nodes, perm)}
    inv = {v: k for k, v in mapping.items()}

    G2 = nx.relabel_nodes(G, mapping, copy=True)
    T2 = nx.minimum_spanning_tree(G2, weight=length_attr, algorithm="kruskal")
    T = nx.relabel_nodes(T2, inv, copy=True)

    T.add_nodes_from(G.nodes(data=True))
    return T

def debug_weight_stats(G: nx.Graph,
                       length_attr: str = "length",
                       weight_attr: str = "weight",
                       prefix: str = ""):
    """
    Print detailed statistics of edge lengths and weights in the graph.
    - Inputs:
        - G: input graph
        - length_attr: edge attribute name for lengths
        - weight_attr: edge attribute name for weights
        - prefix: optional prefix string for log messages
    - Outputs:
        - Prints min, median, mean, max, std, quantiles of lengths and weights
        - Prints count of weights <= 1e-10
    - Procedure:
        - Initialize empty lists Ls and Ws
        - For each edge in G:
            - Append length attribute to Ls
            - Append weight attribute to Ws
        - Convert Ls and Ws to numpy arrays
        - Define helper function qs(x) to compute quantiles
        - Print length statistics
        - Print weight statistics
    """
    Ls = []
    Ws = []
    for _, _, d in G.edges(data=True):
        Ls.append(float(d.get(length_attr, 1.0)))
        Ws.append(float(d.get(weight_attr, 1.0)))

    Ls = np.array(Ls)
    Ws = np.array(Ws)

    def qs(x):
        return np.quantile(x, [0.01, 0.05, 0.5, 0.95, 0.99])

    print(
        f"{prefix}[weight-debug] "
        f"L(min/med/mean/max/std)="
        f"{Ls.min():.4f}/{np.median(Ls):.4f}/{Ls.mean():.4f}/{Ls.max():.4f}/{Ls.std():.4f}",
        flush=True
    )
    print(
        f"{prefix}[weight-debug] "
        f"W(min/med/mean/max/std)="
        f"{Ws.min():.4e}/{np.median(Ws):.4e}/{Ws.mean():.4e}/{Ws.max():.4e}/{Ws.std():.4e}",
        flush=True
    )
    print(
        f"{prefix}[weight-debug] "
        f"W quantiles 1/5/50/95/99% = {qs(Ws)}",
        flush=True
    )
    print(
        f"{prefix}[weight-debug] "
        f"#(W<=1e-10) = {np.sum(Ws <= 1e-10)} / {len(Ws)}",
        flush=True
    )
