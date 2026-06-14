import os
import time
import math
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import networkx as nx
from scipy.sparse import coo_matrix, csr_matrix, triu
from scipy.sparse.csgraph import minimum_spanning_tree
import ot

import joblib
from joblib import Parallel, delayed, parallel_backend
from itertools import product

from tqdm import tqdm
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["font.family"] = "DejaVu Sans"

import sys, pathlib
ROOT = pathlib.Path.cwd() / "orcml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orcml.scripts.pruning import OneDimPruningExperiment, TwoDimPruningExperiment
from orcml.src.experiments.utils.exp_utils import prune_helper
from orcml.src.utils.graph_utils import get_edge_labels
from orcml.src.orcmanl import ORCManL

from dataio import dataio

class AbstractCurvatureCalculator(dataio):
    def __init__(self):
        super().__init__()
        self.X_selected = None
        self.y_selected = None
        self.num_samples = None
        self.kappa_matrix = None
        self.time_duration = None

        self.W = None  # Adjacency matrix
        self.W_pruned = None  # Pruned adjacency matrix

        self.ari = None  # Adjusted Rand Index
        self.ari_pruned = None  # Adjusted Rand Index for pruned graph
        self.labels = None  # Clustering labels
        self.labels_pruned = None  # Clustering labels for pruned graph

        self.process_time = None

    def build_graph(self, 
                    k: int = 10, 
                    symmetric_mode: str = "union"):
        """
        Build kNN graph or full graph.

        Parameters:
            - k : The number of neighbors for defining support set S.
            - symmetric_mode : {"union","mutual"} 
        """
        self.dist_mat = pairwise_distances(self.X_selected, metric="euclidean")

        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.num_samples))

        if k == 0: ## full graph
            iu, ju = np.triu_indices(self.num_samples, k=1)
            self.G.add_weighted_edges_from(
                (int(u), int(v), float(self.dist_mat[u, v])) for u, v in zip(iu, ju)
            )
        else: ## kNN graph
            np.fill_diagonal(self.dist_mat, np.inf)
            knn_idx = np.argpartition(self.dist_mat, kth=k, axis=1)[:, :k]
            np.fill_diagonal(self.dist_mat, 0.0)

            edges = []
            for i in range(self.num_samples):
                for j in knn_idx[i]:
                    if np.isfinite(self.dist_mat[i, j]):
                        u, v = (i, j) if i < j else (j, i)
                        edges.append((u, v, float(self.dist_mat[u, v])))

            if symmetric_mode == "mutual":
                knn_sets = [set(knn_idx[i]) for i in range(self.num_samples)]
                edges = [(u, v, w) for (u, v, w) in edges if (u in knn_sets[v] and v in knn_sets[u])]

            edge_min = {}
            for u, v, w in edges:
                if (u, v) not in edge_min or w < edge_min[(u, v)]:
                    edge_min[(u, v)] = w
            self.G.add_weighted_edges_from((u, v, w) for (u, v), w in edge_min.items())

    def compute_kappa_matrix(self):
        raise NotImplementedError()

class SobolevRicciCurvature(AbstractCurvatureCalculator):
    def __init__(self,
                p: float = 1, 
                sigma: float = 0.1, 
                z0: int = 0,
                sample_kNN: int = 50,
                graph_k: int = 15):
        super().__init__()
        self.p = float(p)
        self.sigma = float(sigma)
        self.z0 = z0

        self.edge_lambdas = {}
        self.gamma_sets = {}
        self.mu_matrix = None  # [n_samples x n_samples] dense
        self.mu_gamma = {}     # dict of [n_samples x n_edges]
        self.sample_kNN = sample_kNN  # The number of neighbors for defining \mu_x
        self.graph_k = graph_k        # The number of neighbors for building kNN graph
        self.edge_list = []
        self.edge_weights = None
        self.edge_num = None

        self.G = nx.Graph()
        self.G_tilde = nx.Graph()

    def _calc_paiwise_distance(self):
        if hasattr(self, "dist_mat") and self.dist_mat is not None:
            print("Distance matrix already calculated. Skipping...")
            return

        self.X_selected = self.X_selected.astype('float32')
        # build MST/SPT with L^p-norm
        print('Started calculating L^p distance matrix.')
        self.dist_mat = pairwise_distances(self.X_selected, metric='minkowski', p=self.p)
        print('End calculating L^p distance matrix.')

    def _ensure_knn_cache(self, 
                          max_neighbors: int) -> None:
        """
        Ensure kNN cache is prepared for up to max_neighbors.
        """
        need = max_neighbors + 1  # Add 1 for self
        has_cache = hasattr(self, "_knn_indices_all") and hasattr(self, "_knn_dists_all") and hasattr(self, "_knn_capacity")
        if has_cache and self._knn_capacity >= need:
            return

        print(f"Preparing kNN cache (n_neighbors={need}) ...")
        data_array = self.X_selected.astype(np.float32, copy=False)

        nbrs = NearestNeighbors(n_neighbors=need, metric='minkowski', p=self.p)
        nbrs.fit(data_array)

        dists_all, indices_all = nbrs.kneighbors(data_array, return_distance=True)
        # Drop self-distances
        self._knn_indices_all = indices_all[:, 1:]  # shape: (n_samples, need-1)
        self._knn_dists_all   = dists_all[:, 1:]
        self._knn_capacity    = need
        print(f' kNN cache ready with capacity {self._knn_capacity}.')

    def build_mst(self):
        '''
        Build Minimum Spanning Tree (MST) using L^p-norm
        '''
        self._calc_paiwise_distance()

        print('Started building MST.')
        # Get upper triangular indices (i<j)
        i_idx, j_idx = np.triu_indices(self.num_samples, k=1)
        # Get weights for these edges
        weights = self.dist_mat[i_idx, j_idx]
        adj_matrix = coo_matrix((weights, (i_idx, j_idx)), shape=(self.num_samples, self.num_samples))
        # Calculate MST with scipy's minimum_spanning_tree
        mst_sparse = minimum_spanning_tree(adj_matrix)

        # Convert to NetworkX graph
        self.G = nx.from_scipy_sparse_array(mst_sparse)
        print('End building MST.')

        # Record \lambda(e) with L^p-norm
        self.edge_lambdas = {
            (u, v) if u <= v else (v, u): data['weight']
            for u, v, data in self.G.edges(data=True)
        }

    def build_spt(self,
                  k=15, 
                  symmetric=True,
                  ) -> None:
        """
        Build a Shortest-Path Tree (SPT) on a kNN graph using L^p distances.

        Parameters:
            - k : The number of neighborhood of kNN Graph
            - symmetric : If True, use mutual kNN graph; otherwise, use directed kNN graph.
        """
        # if full_dist:
        self._calc_paiwise_distance()

        # kNN graph (weights = L^p distances)
        print('Started building kNN graph.')
        # For each i, get the k nearest neighbors excluding the diagonal
        np.fill_diagonal(self.dist_mat, np.inf)
        knn_idx = np.argpartition(self.dist_mat, kth=k, axis=1)[:, :k]
        np.fill_diagonal(self.dist_mat, 0.0)

        G_base = nx.Graph()
        G_base.add_nodes_from(range(self.num_samples))

        if symmetric:
            # Build mutual kNN graph
            edges = []
            for i in range(self.num_samples):
                for j in knn_idx[i]:
                    if i == j:
                        continue
                    w = self.dist_mat[i, j]
                    if np.isfinite(w):
                        u, v = (i, j) if i <= j else (j, i)
                        edges.append((u, v, w))
            # Merge with minimum weight (remove duplicates and ensure bidirectionality)
            # Since the same (u,v) may appear multiple times, take the minimum distance
            edge_min = {}
            for u, v, w in edges:
                key = (u, v)
                if key not in edge_min or w < edge_min[key]:
                    edge_min[key] = w
            for (u, v), w in edge_min.items():
                G_base.add_edge(u, v, weight=float(w))
        else:
            # Build directed kNN graph (as undirected with possible duplicates)
            for i in range(self.num_samples):
                for j in knn_idx[i]:
                    if i == j:
                        continue
                    w = float(self.dist_mat[i, j])
                    if np.isfinite(w):
                        G_base.add_edge(i, j, weight=w)

        print(f'End building kNN graph. |V|={G_base.number_of_nodes()} |E|={G_base.number_of_edges()}')

        # SPT with Dijkstra
        print('Started building SPT with Dijkstra.')
        pred, _ = nx.dijkstra_predecessor_and_distance(G_base, self.z0, weight='weight')
        self.G = nx.Graph()
        self.G.add_nodes_from(G_base.nodes())
        for v in range(self.num_samples):
            if v == self.z0:
                continue
            if v in pred and len(pred[v]) > 0:
                u = pred[v][0]
                # Get the weight from the original distance matrix
                w = float(self.dist_mat[u, v])
                self.G.add_edge(u, v, weight=w)

        print(f'End building SPT. root={self.z0} |E|={self.G.number_of_edges()}')

        # Record \lambda(e) with L^p-norm
        self.edge_lambdas = {
            (u, v) if u <= v else (v, u): data['weight']
            for u, v, data in self.G.edges(data=True)
        }

    def build_gamma_sets(self):
        '''
        For each edge e in the tree, define \gamma_e as the set of nodes whose path from z0 includes e.
        '''
        # Define \gamma_e from the root of path
        self.gamma_sets = defaultdict(set)
        paths = nx.single_source_shortest_path(self.G, source=self.z0)
        for target, path in paths.items():
            for u, v in zip(path[:-1], path[1:]):
                e = (u, v) if u < v else (v, u)
                self.gamma_sets[e].add(target)

    def compute_mu_matrix(self):
        '''
        Compute the neighbor distribution matrix \mu using Gaussian kernel on kNN graph.
        '''
        print("Building kNN graph for mu_x definition.")
        nbrs = NearestNeighbors(n_neighbors=self.sample_kNN + 1, metric='precomputed')
        nbrs.fit(np.zeros((self.num_samples, self.num_samples)))  # dummy fit
        distances, indices = nbrs.kneighbors(self.dist_mat)

        # exclude self (first column)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        self.mu_matrix = np.zeros((self.num_samples, self.num_samples), dtype=np.float32)
        weights = np.exp(-(distances ** 2) / (self.sigma ** 2))
        weights /= weights.sum(axis=1, keepdims=True)

        for row in range(self.num_samples):
            self.mu_matrix[row, indices[row]] = weights[row].astype(np.float32)

    def precompute_mu_gamma(self):
        '''
        Precompute the neighbor distribution over the gamma sets.
        '''
        # For each edge e, precompute sum_j mu[i, j] over gamma_e
        for e, gamma in self.gamma_sets.items():
            gamma_idx = list(gamma)
            self.mu_gamma[e] = self.mu_matrix[:, gamma_idx].sum(axis=1)

    def precompute_static_arrays(self):
        """
        Precompute static arrays for kappa computation.
        """
        # Fix edge order
        self.edge_list = list(self.edge_lambdas.keys())
        self.edge_weights = np.array([self.edge_lambdas[e] for e in self.edge_list], dtype=np.float32)  # (m,)
        self.edge_num = len(self.edge_list)

        self.mu_gamma_matrix = np.stack([self.mu_gamma[e] for e in self.edge_list], axis=1).astype(np.float32)  # (n,m)

        # Incidence matrix of gamma sets
        self.gamma_incidence = np.zeros((self.edge_num, self.num_samples), dtype=np.float32)
        for edge_idx, e in enumerate(self.edge_list):
            if e in self.gamma_sets:
                idx = np.fromiter(self.gamma_sets[e], dtype=np.int64)
                self.gamma_incidence[edge_idx, idx] = 1.0  # (m,n)

    def compute_kappa_matrix(self):
        """
        Compute the curvature matrix \kappa based on Sobolev Ricci Curvature.
        """
        self.kappa_matrix = np.zeros((self.num_samples, self.num_samples), dtype=np.float32)

        for i in tqdm(range(self.num_samples), miniters=10):
            mu_i = self.mu_gamma_matrix[i]  # (m,)
            Gi = self.gamma_incidence[:, i]  # (m,)
            for j in range(i + 1, self.num_samples):
                mu_j = self.mu_gamma_matrix[j]   # (m,)
                Gj   = self.gamma_incidence[:, j]  # (m,)

                st = np.dot(self.edge_weights, np.abs(mu_i - mu_j) ** self.p)
                dj = np.dot(self.edge_weights, np.abs(Gi - Gj))

                self.kappa_matrix[i, j] = self.kappa_matrix[j, i] = 1.0 - (st / dj) ** (1.0 / self.p)

    def build_graph_from_data(self,
                            mode="knn",
                            symmetric="mutual") -> None:
        """
        Build a graph G_tilde from data points using k-NN or full connection.

        Parameters:
            - mode: "knn" for k-NN graph, "full" for complete graph
            - symmetric: "mutual" or "union" for kNN graph symmetry.
        """
        # --- Get kNN indices & distances (without changing the math) ---
        if mode == "full":
            # Avoid O(n^2) copy; temporarily mask diagonal as inf for argpartition
            self._calc_paiwise_distance()
            diag_idx = np.diag_indices(self.num_samples)
            diag_backup = self.dist_mat[diag_idx].copy()
            self.dist_mat[diag_idx] = np.inf  # Exclude self from neighbors

            knn_indices = np.argpartition(self.dist_mat, kth=self.graph_k, axis=1)[:, :self.graph_k]
            knn_dists   = np.take_along_axis(self.dist_mat, knn_indices, axis=1)

            self.dist_mat[diag_idx] = diag_backup  # Restore diagonal
        else:
            self._ensure_knn_cache(self.graph_k)
            knn_indices = self._knn_indices_all[:, :self.graph_k]
            knn_dists   = self._knn_dists_all[:,   :self.graph_k]

        # Flatten neighbor structure
        rows = np.repeat(np.arange(self.num_samples), self.graph_k)        # source nodes (i)
        cols = knn_indices.ravel()               # neighbor nodes (j)
        vals = knn_dists.ravel().astype(float)   # weights w_ij

        # Remove self-loops (keep graph semantics identical)
        mask = rows != cols
        rows, cols, vals = rows[mask], cols[mask], vals[mask]

        # Convert to undirected keys (u, v) with u < v
        u = np.minimum(rows, cols)
        v = np.maximum(rows, cols)

        if symmetric == "union":
            # Take the minimum weight across all directed occurrences of an undirected edge
            order = np.lexsort((v, u))
            u, v, vals = u[order], v[order], vals[order]

            # Segment starts for each unique (u, v)
            changed = (np.diff(u) != 0) | (np.diff(v) != 0)
            starts = np.concatenate(([0], np.where(changed)[0] + 1))
            mins = np.minimum.reduceat(vals, starts)

            u_unique = u[starts]
            v_unique = v[starts]

            self.G_tilde = nx.Graph()
            self.G_tilde.add_nodes_from(range(self.num_samples))
            self.G_tilde.add_weighted_edges_from(
                zip(u_unique.tolist(), v_unique.tolist(), mins.tolist())
            )
        else:
            # --- symmetric == "mutual" ---
            # Prefer a sparse path when SciPy is available (fast and memory-efficient)
            # Boolean adjacency for directed kNN
            A = csr_matrix((np.ones_like(vals, dtype=bool), (rows, cols)), shape=(self.num_samples, self.num_samples))
            # Weight matrix aligned with A
            W = csr_matrix((vals, (rows, cols)), shape=(self.num_samples, self.num_samples))

            # Mutual mask: neighbors in both directions
            mut_mask = A.multiply(A.T)

            # For mutual edges, take min of two directed weights
            Wmin = W.minimum(W.T)

            # Keep only upper triangular to avoid duplicates (u < v)
            mutual_weights = triu(Wmin.multiply(mut_mask), k=1).tocsr()
            I, J = mutual_weights.nonzero()
            weights = mutual_weights.data

            self.G_tilde = nx.Graph()
            self.G_tilde.add_nodes_from(range(self.num_samples))
            self.G_tilde.add_weighted_edges_from(zip(I.tolist(), J.tolist(), weights.tolist()))

    def compute_kappa_matrix_in_Graph_vectorized(self, 
                                                 G: nx.Graph):
        '''
        Compute the curvature matrix \kappa for all edges in graph G using vectorized operations.

        Parameters:
            - G: The graph for which to compute the curvature matrix.
        '''
        lambdas = self.edge_weights.astype(np.float32)           # (m,)
        M = self.mu_gamma_matrix.astype(np.float32)              # (n,m)
        Gamma = self.gamma_incidence.astype(np.float32).T        # (n,m)
        self.kappa_matrix = np.zeros((self.num_samples, self.num_samples), dtype=np.float32)

        # Initialize kappa matrix
        edges = np.array([(min(i,j), max(i,j)) for (i,j) in G.edges()], dtype=np.int64)
        I, J = edges[:,0], edges[:,1]

        # S_p^p(i,j) = ⟨λ, | M[i,:] − M[j,:] |^p ⟩
        Diff_M = M[I, :] - M[J, :]                               # (E, m)
        p = getattr(self, 'p', 1)
        if p == 1:
            Sp_p = np.abs(Diff_M) @ lambdas                      # (E,)
        elif p == 2:
            Sp_p = (Diff_M * Diff_M) @ lambdas
        else:
            Sp_p = (np.abs(Diff_M) ** p) @ lambdas
        Sp = np.power(np.maximum(Sp_p, 0.0), 1.0 / p)

        # d_tree(i,j) = ⟨λ, | Γ[i,:] − Γ[j,:] | ⟩
        Diff_G = Gamma[I, :] - Gamma[J, :]                       # (E, m)
        d_tree = np.abs(Diff_G) @ lambdas                        # (E,)
        d_tree = np.maximum(d_tree, 1e-12)

        # Fill the kappa matrix
        self.kappa_matrix[I, J] = self.kappa_matrix[J, I] = 1.0 - (Sp / d_tree)

    def compute_S2_over_c_full(self):
        '''
        Use for getting S2/c matrix and use it for kernel method.
        '''
        lambdas = self.edge_weights.astype(np.float32)          # (m,)
        M = self.mu_gamma_matrix.astype(np.float32)             # (n,m)
        Gamma = self.gamma_incidence.astype(np.float32).T       # (n,m)  # node x edge (γ_e)

        I, J = np.triu_indices(self.num_samples, k=1)

        Diff_M = M[I, :] - M[J, :]                              # (E, m)
        S2_sq = (Diff_M * Diff_M) @ lambdas                     # (E,)
        S2 = np.sqrt(np.maximum(S2_sq, 0.0))                    # (E,)

        Diff_G = Gamma[I, :] - Gamma[J, :]                      # (E, m)
        D2_sqsum = np.abs(Diff_G) @ lambdas                     # (E,)  == Σ_path λ(e)
        D2 = np.sqrt(np.maximum(D2_sqsum, 1e-12))               # (E,)

        S2_over_c = S2 / D2.mean()
        self.S2_over_c_matrix = np.zeros((self.num_samples, self.num_samples), dtype=np.float32)
        self.S2_over_c_matrix[I, J] = self.S2_over_c_matrix[J, I] = S2_over_c
        np.fill_diagonal(self.S2_over_c_matrix, 0.0)

class JudgeEdges():
    def __init__(self,
                 dataset: str,
                 n_points: int = 2000,
                 two_dim: bool = False,):
        self.dataset = dataset
        self.n_points = n_points

        self.exp = None
        self.data = None
        self.cluster = None
        self.good_edges = None
        self.shortcut_edges = None
        self.G = None
        if two_dim:
            self.exp = TwoDimPruningExperiment()
        else:
            self.exp = OneDimPruningExperiment()
        self.exp_params = None
        self.df = pd.DataFrame()

        ## For edge pruning threshold
        self.kappa_thresh = None
        self.delta_M  = None
        self.lambda_M = None
        self.epsilon = None

    def split_edge_labels_flexible(self, 
                                   G: nx.Graph, 
                                   edge_labels):
        """
        Split edges into 'good' and 'shortcut' based on flexible input formats.

        Parameters:
            - G: networkx.Graph
            - edge_labels: Can be one of the following formats:
                1. dict with keys as (i,j) tuples and values as labels
                2. list/tuple/ndarray of (i,j,label) tuples
                3. list/tuple/ndarray of ((i,j), label) tuples
                4. list/tuple/ndarray of labels aligned with G.edges()
        """
        def canon(u,v):
            u, v = int(u), int(v)
            return (u, v) if u < v else (v, u)

        good, short = set(), set()

        def add(u, v, lab):
            if isinstance(lab, (int, float, np.integer, np.floating)):
                if lab > 0:
                    good.add(canon(u, v))
                else:
                    short.add(canon(u, v))
            else:
                lab_s = str(lab).lower()
                if lab_s in {"good", "1", "true"}:
                    good.add(canon(u, v))
                else:
                    short.add(canon(u, v))

        if hasattr(edge_labels, "items"):
            for (i, j), lab in edge_labels.items():
                add(i, j, lab)

        elif isinstance(edge_labels, (list, tuple, np.ndarray)):
            if len(edge_labels) == 0:
                return [], []

            e0 = edge_labels[0]
            # (i,j,lab)-format
            if isinstance(e0, (list, tuple)) and len(e0) == 3:
                for i, j, lab in edge_labels:
                    add(i, j, lab)
            # ((i,j), lab)-format
            elif isinstance(e0, (list, tuple)) and len(e0) == 2 and isinstance(e0[0], (list, tuple)):
                for (i, j), lab in edge_labels:
                    add(i, j, lab)
            else:
                # label-list format aligned with G.edges()
                edges_list = list(G.edges())
                if len(edge_labels) != len(edges_list):
                    raise ValueError(
                        f"edge_labels has length {len(edge_labels)} but G has {len(edges_list)} edges; "
                        "provide labels aligned with G.edges()."
                    )
                for (i, j), lab in zip(edges_list, edge_labels):
                    add(i, j, lab)
        else:
            raise TypeError(f"Unsupported edge_labels type: {type(edge_labels)}")

        return sorted(good), sorted(short)

    def generate_data_labels(self):
        '''
        Generate data and labels, construct kNN graph, and split edges into good and shortcut.
        '''
        self.exp_params = self.exp.param_map[self.dataset]

        self.shortcut_edges = []
        while len(self.shortcut_edges) <= 0:
            ## Generating data
            self.data, self.cluster, data_supersample, subsample_indices, dataset_info = self.exp.map[self.dataset](n_points=self.n_points)

            ## Construct kNN-graph
            return_dict = prune_helper(self.data, exp_params=self.exp_params)
            self.G = return_dict['G_original']

            ## Edge labels
            edge_labels = get_edge_labels(
                G=self.G,
                cluster=self.cluster,
                data_supersample_dict={'data_supersample': data_supersample,
                                    'subsample_indices': subsample_indices},
                scale=self.exp_params['edge_label_est_scale']
            )

            ## Evaluate the generated data
            self.good_edges, self.shortcut_edges = self.split_edge_labels_flexible(self.G, edge_labels)
            print(f"Good edges: {len(self.good_edges)}, Shortcut edges: {len(self.shortcut_edges)}")
    
    def _canon_edge(self,u, v):
        '''
        Normalize edge (u,v) to (min(u,v), max(u,v)).
        '''
        return (u, v) if u < v else (v, u)

    def _to_edge_set(self, edges_like) -> set:
        """
        Convert edges_like to a set of canonical edge tuples (i,j) with i<j.

        Parameters:
            - edges_like: Can be a networkx.Graph, or an iterable of edge tuples.
        
        Returns:
            - Set of tuples (i,j) with i<j.
        """
        S = set()
        if hasattr(edges_like, "edges"):
            iterator = edges_like.edges()
        else:
            iterator = edges_like
        for e in iterator:
            i, j = int(e[0]), int(e[1])
            if i != j:
                S.add(self._canon_edge(i, j))
        return S

    def estimate_epsilon_from_graph(self, G: nx.Graph) -> float:
        """
        Estimate epsilon as the 90th percentile of edge weights in G.

        Parameters:
            - G: networkx.Graph with 'weight' attribute on edges.
        
        Returns:
            - Estimated epsilon value.
        """
        ws = [d.get("weight", 1.0) for *_ , d in G.edges(data=True)]
        return float(np.percentile(ws, 90)) if len(ws) else 1.0

    def orcmanl_prune_edges_from_kappa(self,
                                    kappa_matrix: np.ndarray,
                                    ) -> set:
        """
        MANL pruning with curvature threshold and direction.
        - direction: "le" -> prune low-curvature; "ge" -> prune high-curvature
        """
        timing_manl = None
        start = time.time()
        # Candidate set C (respect direction)
        C = self.prune_curvature_only(kappa_matrix=kappa_matrix)

        # Build G' = (V, E \ C)
        Gp = nx.Graph()
        Gp.add_nodes_from(self.G.nodes())
        for u, v, d in self.G.edges(data=True):
            if self._canon_edge(u, v) not in C:
                Gp.add_edge(u, v, **d)

        # Shortest-path threshold T
        T = (math.pi*(math.pi+1.0)*(1.0 - self.lambda_M)) / (2.0*math.sqrt(24.0*self.lambda_M)) * self.epsilon

        # Final MANL pruning (subset of C)
        pruned = set()
        for (i, j) in C:
            try:
                dsp = nx.shortest_path_length(Gp, source=i, target=j, weight="weight")
            except nx.NetworkXNoPath:
                dsp = float("inf")
            if dsp > T:
                pruned.add((i, j))
        timing_manl = time.time() - start
        return pruned, timing_manl

    def prune_curvature_only(self,
                            kappa_matrix: np.ndarray,) -> set:
        """
        Prune with Curvature \kappa only
        """
        P = set()
        for u, v in self.G.edges():
            i, j = self._canon_edge(u, v)
            if np.isfinite(kappa_matrix[i,j]) and kappa_matrix[i,j] <= self.kappa_thresh:
                P.add((i, j))
        return P

    def prune_distance(self,
                       mult: float = 1.0) -> set:
        """
        Prune edges longer than mult * epsilon.

        Parameters:
            - epsilon: Scale parameter; if None, it will be estimated from G.
            - mult: Multiplier for epsilon to set the threshold.
        
        Returns:
            - Set of edges (i,j) to be pruned.
        """
        thr = mult * self.epsilon
        P = set()
        for u, v, d in self.G.edges(data=True):
            if float(d.get("weight", 1.0)) > thr:
                P.add(self._canon_edge(u, v))
        return P

    def _bootstrap_removed_pct(self,
                            pruned: set,
                            subset: set,
                            n_boot: int = 500,
                            seed: int = 0,
                            ) -> tuple:
        """
        Bootstrap resampling to estimate the mean and std of the removal percentage.

        Parameters:
            - pruned: Set of edges that were pruned.
            - subset: Set of edges to evaluate (good or shortcut).
            - n_boot: Number of bootstrap samples.
            - seed: Random seed for reproducibility.
        
        Returns:
            - Tuple of (mean removal percentage, std of removal percentage).
        """
        if len(subset) == 0:
            return 0.0, 0.0
        rng = np.random.default_rng(seed)
        arr = np.array(list(subset), dtype=int)
        m = len(arr)
        vals = np.empty(n_boot, float)
        for b in range(n_boot):
            idx = rng.integers(0, m, size=m)
            sample = arr[idx]
            cnt = sum((self._canon_edge(i,j) in pruned) for (i,j) in sample)
            vals[b] = 100.0 * cnt / m
        return float(vals.mean()), float(vals.std(ddof=1))

    def calc_curvatures(self,
                        sc_spt_kNN: int = 15):
        '''
        Calculate ORC and SC curvature matrices.
        Parameters:
            - sc_spt_kNN: kNN parameter for SC(SPT) construction.
        '''
        timing = {}
        ## Calculate ORC
        print('Start calculating ORC')

        start = time.time()
        orc = ORCManL(exp_params=self.exp.param_map[self.dataset], verbose=False)
        orc.fit(self.data)

        ## Convert to np.ndarray
        orc_kappa = np.zeros((self.n_points, self.n_points), dtype=float)
        for u, v, d in orc.G_pruned.edges(data=True):
            orc_kappa[u, v] = d["ricciCurvature"]

        print(f"ORC calculation completed in {time.time() - start:.2f} seconds.")
        timing["orc"] = time.time() - start

        ## Calculate SC
        print('Start calculating SC')
        sc1 = SobolevRicciCurvature(p=2, sigma=2.0, z0=0, 
                                          sample_kNN=50)
        sc1.X_selected = self.data
        sc1.num_samples = sc1.X_selected.shape[0]

        start = time.time()
        sc1.build_spt(k=sc_spt_kNN)
        sc1.build_gamma_sets()
        sc1.compute_mu_matrix()
        sc1.precompute_mu_gamma()
        sc1.precompute_static_arrays()
        sc1.compute_kappa_matrix_in_Graph_vectorized(G=orc.G_pruned)
        print(f"SC calculation completed in {time.time() - start:.2f} seconds.")
        timing["src"] = time.time() - start

        return orc_kappa, sc1.kappa_matrix, timing

    def pruning_report_table_bootstrap_all(
            self,
            src_kappa: np.ndarray,
            orc_kappa: np.ndarray,
            delta: float = 0.8,
            lam: float = 0.01,
            epsilon: float = 0.01,
            n_boot: int = 1000,
            seed: int = 0,
            dist_mult: float = 1.0,
            curvature_time: dict = None,
        ) -> pd.DataFrame:
        # =========================
        # Setup
        # =========================
        if epsilon is None:
            self.epsilon = self.estimate_epsilon_from_graph(self.G)
        else:
            self.epsilon = epsilon

        good_set  = self._to_edge_set(self.good_edges)
        short_set = self._to_edge_set(self.shortcut_edges)

        self.delta_M = delta
        self.kappa_thresh = -1.0 + 4.0 * (1.0 - self.delta_M)
        self.lambda_M = lam

        print(f"Pruning parameters: delta_M={self.delta_M}, lambda_M={self.lambda_M}, "
            f"kappa_thresh={self.kappa_thresh:.4f}, epsilon={self.epsilon:.4f}")

        # =========================
        # Helper: timing wrapper
        # =========================
        timing_results = {}

        def run_with_timing(name, func):
            t0 = time.time()
            result = func()
            elapsed = time.time() - t0
            timing_results[name] = elapsed
            return result

        # =========================
        # Method definitions
        # =========================
        method_configs = {
            "ORC-MANL": lambda: self.orcmanl_prune_edges_from_kappa(orc_kappa)[0],
            "SRC-MANL": lambda: self.orcmanl_prune_edges_from_kappa(src_kappa)[0],
            "ORC only": lambda: self.prune_curvature_only(kappa_matrix=orc_kappa),
            "SRC only": lambda: self.prune_curvature_only(kappa_matrix=src_kappa),
            "Distance": lambda: self.prune_distance(mult=dist_mult),
        }

        # =========================
        # Run methods
        # =========================
        methods = {}
        for name, func in method_configs.items():
            methods[name] = run_with_timing(name, func)

        # =========================
        # Bootstrap evaluation
        # =========================
        top_mean, top_std = {}, {}
        bottom_mean, bottom_std = {}, {}

        for name, P in methods.items():
            g_mean, g_std = self._bootstrap_removed_pct(P, good_set,  n_boot=n_boot, seed=seed)
            s_mean, s_std = self._bootstrap_removed_pct(P, short_set, n_boot=n_boot, seed=seed+1)

            top_mean[name], top_std[name] = g_mean, g_std
            bottom_mean[name], bottom_std[name] = s_mean, s_std

        # =========================
        # Time aggregation
        # =========================
        def get_time_curvature(name):
            if curvature_time is None:
                return 0.0
            if "ORC" in name:
                return curvature_time.get("orc", 0.0)
            elif "SRC" in name:
                return curvature_time.get("src", 0.0)
            return 0.0

        def get_time_total(name):
            return timing_results.get(name, 0.0) + get_time_curvature(name)

        # =========================
        # Build DataFrame
        # =========================
        rows = []
        for name in methods.keys():
            for metric_name, mean_dict, std_dict in [
                ("good removed (%)", top_mean, top_std),
                ("shortcut removed (%)", bottom_mean, bottom_std),
            ]:
                rows.append({
                    "method": name,
                    "metric": metric_name,
                    "mean": float(mean_dict[name]),
                    "std": float(std_dict[name]),

                    # --- timing ---
                    "time_pruning": timing_results.get(name, 0.0),
                    "time_curvature": get_time_curvature(name),
                    "time_total": get_time_total(name),

                    # --- meta ---
                    "dataset": self.dataset,
                    "delta": self.delta_M,
                    "lambda": self.lambda_M,
                })

        self.df = pd.DataFrame(rows)
        return self.df

def _worker_edge_pruning_safe(delta: float,
                              lam: float,
                              n_points: int,
                              datasets: list,
                              datasets_2dim: list,
                              need_data: bool = False) -> tuple:
    """
    Run one (delta, lam) job and return (ok, df, delta, lam, err).
    Using a safe wrapper allows us to keep the joblib loop simple
    without crashing the whole run on a single failure.
    """
    try:
        util = EdgePruningUtil()
        df = util.apply_calc_edge_removal(
            datasets=datasets,
            datasets_2dim=datasets_2dim,
            n_points=n_points,
            delta=delta,
            lam=lam,
            need_data=need_data
        )
        df["delta"] = delta
        df["lam"] = lam
        return True, df, delta, lam, None
    except Exception as e:
        return False, None, delta, lam, e

class EdgePruningUtil():
    '''
    Utility class for edge pruning operations.
    '''
    def __init__(self):
        self.df_all = None

    def calc_edge_removal(self,
                            dataset: str,
                            n_points: int,
                            two_dim: bool,
                            delta: float = 0.8,
                            lam: float = 0.01,
                            need_data: bool = True) -> JudgeEdges:
        '''
        Calculate edge removal statistics for a given dataset.

        Parameters:
            - dataset: Name of the dataset.
            - n_points: Number of data points to generate.
            - two_dim: Whether to use 2D data generation.
            - delta: Curvature threshold for edge pruning.
            - lam: Parameter for edge pruning.
            - need_data: Whether to generate new data or load existing data.

        Returns:
            - JudgeEdges instance with results.
        '''
        print(f'Start {dataset}')
        ## if need_data, generate new data; else load existing data
        if need_data:
            shortcut_edge = 0
            while shortcut_edge <= 0:
                je = JudgeEdges(dataset=dataset,
                                n_points=n_points,
                                two_dim=two_dim)
                je.generate_data_labels()
                shortcut_edge = len(je.shortcut_edges)
                print(f"shortcut edges: {shortcut_edge}")
            joblib.dump(je, f'./data/je_{dataset}_n{n_points}_2d{two_dim}.pkl')
        ## Load existing data
        else:
            ## Save dataset
            je = joblib.load(f'./data/je_{dataset}_n{n_points}_2d{two_dim}.pkl')

        # compute curvatures
        orc_kappa, src_kappa, curv_process_time = je.calc_curvatures(sc_spt_kNN=50)

        # baseline pruning report
        je.pruning_report_table_bootstrap_all(
            orc_kappa=orc_kappa,
            src_kappa=src_kappa,
            delta=delta, lam=lam, epsilon=None,
            n_boot=10, seed=0, dist_mult=1.0,
            curvature_time=curv_process_time
        )
        return je

    def apply_calc_edge_removal(self,
                                datasets: list, 
                                datasets_2dim: list,
                                n_points: int = 4000,
                                delta: float = 0.75,
                                lam: float = 0.01,
                                need_data: bool = True) -> pd.DataFrame:
        '''
        Apply edge removal calculation to multiple datasets.

        Parameters:
            - datasets: List of dataset names for 1D data.
            - datasets_2dim: List of dataset names for 2D data.
            - n_points: Number of data points to generate.
            - delta: Curvature threshold for edge pruning.
            - lam: Parameter for edge pruning.
            - need_data: Whether to generate new data or load existing data.
        
        Returns:
            - pd.DataFrame with concatenated results from all datasets.
        '''
        df_tmp = []
        for ds in datasets:
            je = self.calc_edge_removal(dataset=ds,
                            n_points=n_points,
                            two_dim=False,
                            delta=delta, lam=lam,
                            need_data=need_data)
            df_tmp.append(je.df)

        for ds in datasets_2dim:
            je = self.calc_edge_removal(dataset=ds,
                            n_points=n_points,
                            two_dim=True,
                            delta=delta, lam=lam,
                            need_data=need_data)
            df_tmp.append(je.df)
        ## Concat DataFrames
        return pd.concat(df_tmp, axis=0)

    def grid_search_joblib(self,
                           deltas: list,
                           lams: list,
                           datasets: list,
                           datasets_2dim: list,
                           n_points: int = 4000,
                           n_jobs: int | None = None,
                           pre_dispatch: str = "2*n_jobs",
                           batch_size: str | int = "auto",
                           verbose: int = 10,
                           need_data: bool = False) -> pd.DataFrame:
        """
        Parallel grid search over (delta, lam) using joblib (loky backend).
        - Uses a top-level safe worker to avoid pickling issues.
        - Collects errors per-task instead of stopping the whole run.
        """
        # Decide parallelism (default: CPU cores - 1)
        if n_jobs is None:
            cpu = os.cpu_count() or 2
            n_jobs = max(1, cpu - 1)

        combos = list(product(deltas, lams))

        # Use loky (processes). If you are CPU-bound this is preferred.
        # If your workload is I/O-bound or releases the GIL heavily, you can try backend="threading".
        with parallel_backend("loky"):
            results = Parallel(
                n_jobs=n_jobs,
                prefer="processes",     # ensure processes, not threads
                verbose=verbose,
                pre_dispatch=pre_dispatch,
                batch_size=batch_size,
            )(
                delayed(_worker_edge_pruning_safe)(d, l, n_points, datasets, datasets_2dim, need_data=need_data)
                for d, l in combos
            )

        # Split successes and failures
        ok_frames = []
        errors = []
        for ok, df, d, l, err in results:
            if ok:
                ok_frames.append(df)
                print(f"[done] delta={d}, lam={l}, rows={len(df)}")
            else:
                errors.append((d, l, err))
                print(f"[error] delta={d}, lam={l} -> {err}")

        if not ok_frames:
            raise RuntimeError("All jobs failed. See printed errors above.")

        # Optional: surface errors to caller while still returning data
        if errors:
            print(f"Completed with {len(errors)} failed combos.")

        return pd.concat(ok_frames, axis=0, ignore_index=True)

    @staticmethod
    def aggregate_df(df_result: pd.DataFrame,
                     method: str = 'ORC-MANL') -> pd.DataFrame:
        '''
        Aggregate results for a specific method.

        Parameters:
            - df_result: DataFrame with columns ['delta', 'lambda', 'method', 'metric', 'mean', 'std']
            - method: Method name to filter and aggregate.
        Returns:
            - Aggregated DataFrame with columns ['delta', 'lambda', 'method', 'good_removed', 'shortcut_removed', 'effective_removed']
        '''
        # Extract method-specific data
        df_f = df_result[df_result["method"] == method].copy()

        agg = (
            df_f.groupby(["delta", "lambda", "method", "metric"], as_index=False)["mean"]
                .sum()
        )

        wide = (
            agg.pivot(index=["delta", "lambda", "method"], columns="metric", values="mean")
            .reset_index()
        )

        return (
            wide.rename(columns={"good removed (%)": "good_removed",
                                "shortcut removed (%)": "shortcut_removed"})
                .assign(effective_removed=lambda df: df["shortcut_removed"] - df["good_removed"])
                .sort_values("effective_removed", ascending=False)
                .reset_index(drop=True)
        )
