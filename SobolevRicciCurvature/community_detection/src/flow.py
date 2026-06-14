import time 
import math
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from src.curvature import _neighbors_arrays, _edge_index_arrays, _build_tree_parent, _calculate_src_curvature
from src.utils import ensure_edge_lengths

from .GraphRicciCurvature.OllivierRicci import OllivierRicci

class FlowMethod(ABC):
    '''
    Abstract base class for Ricci flow methods.
    '''
    @abstractmethod
    def run(self, G0: nx.Graph, *, 
            seed: int | None = None) -> tuple[nx.Graph, dict]:
        pass

class RicciFlowRunner:
    '''
    Docstring for RicciFlowRunner
    '''
    def __init__(self, method: FlowMethod):
        self.method = method

    def run(self, G0: nx.Graph, *, seed: int | None = None):
        return self.method.run(G0, seed=seed)

class ORCFlowMethod(FlowMethod):
    '''
    Ollivier-Ricci flow method.
    - Inputs:
        - alpha: float
        - steps: int
        - orc_eta: float
        - max_iter: int
    - Outputs:
        - Gf: nx.Graph
        - stats: dict
    - Procedure:
        1. Initialize OllivierRicci with parameters.
        2. Compute Ricci flow for specified steps and eta.
        3. Return the modified graph and statistics.
    '''
    def __init__(self, 
                 alpha: float, 
                 steps: int, 
                 orc_eta: float, 
                 max_iter: int = 200):
        self.alpha = alpha
        self.steps = steps
        self.orc_eta = orc_eta
        self.max_iter = max_iter

    def run(self, 
            G_in: nx.Graph,
            seed: int | None = None):
        Gf, st = self.orc_ricci_flow_equiv(G_in=G_in, 
                                            verbose="INFO",)
        return Gf, st

    def orc_ricci_flow_equiv(self,
                            G_in: nx.Graph,
                            length_attr: str = "length",
                            weight_attr: str = "weight",
                            delta_stop: float = 1e-4,
                            verbose: str = "INFO",
                            ):
        """
        Ollivier-Ricci flow (equivalent measure version).
        """
        t0 = time.perf_counter()

        G = nx.Graph(G_in).copy()

        for u, v, d in G.edges(data=True):
            w = float(d.get(length_attr, 1.0))
            G[u][v][weight_attr] = w

        orc = OllivierRicci(
            G,
            weight=weight_attr,
            alpha=self.alpha,
            method="OTDSinkhornMix",
            base=math.e,
            exp_power=2,
            shortest_path="all_pairs",
            verbose=verbose,
        )
        orc.compute_ricci_flow(iterations=self.steps, step=self.orc_eta, delta=delta_stop)

        Gf = orc.G

        for u, v, d in Gf.edges(data=True):
            d[length_attr] = float(d.get(weight_attr, 1.0))

        t1 = time.perf_counter()
        return Gf, {"time_flow_sec": float(t1 - t0)}

class SRCFlowMethod(FlowMethod):
    '''
    SRC Ricci flow method.
    - Inputs:
        - delta: float
        - steps: int
        - p: float
        - method: str
        - src_root: int
    - Outputs:
        - Gf: nx.Graph
        - stats: dict
    - Procedure:
        1. Initialize parameters.
        2. Compute SRC Ricci flow for specified steps.
        3. Return the modified graph and statistics.
    '''
    def __init__(self, 
                 delta: float, 
                 steps: int, 
                 p: float = 1.0,
                 method: str = "src-spt",
                 src_root: int = 0):
        self.delta = delta
        self.steps = steps
        self.p = p
        self.method = method
        self.n = None
        self.src_root = src_root
        self.epsilon = 1e-4
        
    def run(self, G_in, seed=None):
        '''
        Run SRC Ricci flow method.
        '''
        self.n = G_in.number_of_nodes()
        Gf, st = self.src_ricci_flow(G_in)
        return Gf, st

    def src_ricci_flow(self,
                        G_in: nx.Graph,
                        length_attr: str = "length",
                        epsilon: float = 1e-4,
                    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        SRC Ricci flow
        - Inputs:
            - G_in: nx.Graph
            - length_attr: str
        - Outputs:
            - G: nx.Graph
            - stats: dict
        - Procedure:
            1. Initialize parameters and copy graph.
            2. For each step:
                a. Build shortest path tree (SPT).
                b. Calculate source curvature.
                c. Update edge lengths.
            3. Return modified graph and statistics.
        """

        t0 = time.perf_counter()

        G = G_in.copy()
        ensure_edge_lengths(G, default_len=1.0, attr=length_attr)

        nodes = list(G.nodes())
        p = float(self.p)
        delta = float(self.delta)
        eps = float(self.epsilon)

        # choose root
        src_root = int(self.src_root)
        if src_root not in G:
            src_root = nodes[0]

        # neighbor cache (recomputed if surgery changes topology)
        neigh_idx_list, deg = _neighbors_arrays(G, nodes)

        weight_hist = []
        kappa_hist  = []

        # edge arrays (recomputed if surgery changes topology)
        u_idx, v_idx, L = _edge_index_arrays(G, nodes, length_attr=length_attr)

        prev_kappas = None
        for step in range(int(self.steps)):
            if G.number_of_edges() == 0:
                break

            # --- build SPT on current lengths ---
            # order_nodes, parent, plen = _build_spt_parent(G, root, length_attr=length_attr)
            order_nodes, parent, plen = _build_tree_parent(G, src_root, method=self.method, length_attr=length_attr)

            kappas, sps, dps, E = _calculate_src_curvature(
                parent=parent,
                plen=plen,
                method=self.method,
                step=step,
                n=self.n,
                neigh_idx_list=neigh_idx_list,
                deg=deg,
                delta=delta,
                L=L,
                u_idx=u_idx,
                v_idx=v_idx,
                p=p
            )

            if prev_kappas is not None and len(kappas) == len(prev_kappas):
                diff = float(np.max(np.abs(kappas - prev_kappas)))
                print(f"[SRC-Flow] Step {step}: max kappa diff = {diff:.6e}")
                if diff < eps:
                    print(f"[SRC-Flow] Converged at step {step} (max kappa change {diff:.6e} < eps {eps:.6e})")
                    break

            prev_kappas = kappas.copy()
            weight_hist.append(L.copy())
            kappa_hist.append(kappas.copy())

            # --- paper-consistent update ---
            L = (1.0 - kappas) * L.copy()

            for k in range(E):
                a = nodes[int(u_idx[k])]
                b = nodes[int(v_idx[k])]
                if G.has_edge(a, b):
                    G[a][b][length_attr] = float(L[k])

        t1 = time.perf_counter()
        stats = {
            "time_flow_sec": float(t1 - t0),
            "edges_final": int(G.number_of_edges()),
        }
        stats["weight_hist"] = np.stack(weight_hist, axis=0)
        stats["kappa_hist"]  = np.stack(kappa_hist, axis=0)
        stats["u_idx"] = u_idx.copy()
        stats["v_idx"] = v_idx.copy()
        stats["nodes"] = nodes.copy()
        stats['root'] = src_root

        return G, stats

