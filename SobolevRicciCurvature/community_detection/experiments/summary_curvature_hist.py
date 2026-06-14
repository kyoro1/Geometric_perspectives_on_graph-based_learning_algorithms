import os
import sys
import time
import argparse
import networkx as nx
from typing import Dict, Any

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets import DataLoader
from src.utils import ensure_edge_lengths
from src.methods import make_method
from src.flow import RicciFlowRunner

import numpy as np
import matplotlib.pyplot as plt

label_map = {
    "src-spt": "SRC(SPT)",
    "src-mst": "SRC(MST)",
    "orc": "ORC",
}

def run(lfr_nodes: int,
        method: str,
        mu: float,
        seed: int,
        steps: int,
        delta_or_alpha: float,
        method_kwargs: Dict[str, Any] = None,
        ) -> Dict[str, Any]:

    print('Starting:', method, 'mu=', mu)

    LFR = DataLoader(n=lfr_nodes, seed=seed)
    G0, gt = LFR.generate_lfr(mu=mu, 
                        avg_deg=20,
                        seed_tries=200,
                        tau1=3.0, tau2=1.5,
                        min_community=20, 
                        max_community=100,
                        max_degree=50)

    ensure_edge_lengths(G0, default_len=1.0, attr="length")

    method_obj = make_method(method, 
                             steps, 
                             delta_or_alpha, 
                             method_kwargs=method_kwargs,
                             )

    runner = RicciFlowRunner(method_obj)
    Gf, st = runner.run(G0, seed=seed)

    if method == "src-spt" or method == "src-mst":
        kappa_hist = st["kappa_hist"]
    else:
        kappa_hist = nx.get_edge_attributes(Gf, "ricciCurvature")

    return {
        "method": method,
        "mu": float(mu),
        "seed": int(seed),
        "kappa_hist": kappa_hist
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfr_nodes", type=int, default=500)
    ap.add_argument("--out", type=str, default="lfr_src.csv")
    ap.add_argument("--methods", type=str, default="src-spt,src-mst,orc", help="comma-separated: src-spt,src-mst,orc")
    ap.add_argument("--mus", type=str, default="0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ## For SRC params
    ap.add_argument("--src_p", type=float, default=1.0, help="SRC p parameter")
    ap.add_argument("--delta", type=float, default=0.5, help="SRC delta / ORC alpha")
    args = ap.parse_args()

    methods = [s.strip().lower() for s in args.methods.split(",") if s.strip()]

    kappa_list = []
    for method in methods:
        results = run(
            lfr_nodes=args.lfr_nodes,
            method=method,
            mu=float(args.mus.split(",")[0]),
            seed=args.seed,
            steps=args.steps,
            delta_or_alpha=args.delta,
            method_kwargs={"p": args.src_p},
        )
        kappa_list.append(results["kappa_hist"])

    all_vals = []
    method_vals = {}

    for kappa_hist, method in zip(kappa_list, methods):
        if method in ["src-spt", "src-mst"]:
            vals = kappa_hist.ravel()
        else:
            vals = np.array(list(kappa_hist.values()))
        method_vals[method] = vals
        all_vals.append(vals)

    all_vals = np.concatenate(all_vals)

    bin_edges = np.histogram_bin_edges(all_vals, bins=150)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    color_map = {
        "orc": "C0",      # blue
        "src-mst": "C1",  # orange
        "src-spt": "C2",  # green
    }

    for method in ["src-spt", "src-mst"]:
        if method in method_vals:
            ax_top.hist(
                method_vals[method],
                bins=bin_edges,
                alpha=0.6,
                histtype="stepfilled",
                label=label_map.get(method, method),
                color=color_map[method],
            )

    ax_top.set_title(f"SRC")
    ax_top.legend(loc='upper left', fontsize=14)

    if "orc" in method_vals:
        ax_bottom.hist(
            method_vals["orc"],
            bins=bin_edges,
            alpha=0.6,
            histtype="stepfilled",
            label=label_map.get("orc", "orc"),
            color=color_map["orc"],
        )

    ax_bottom.set_title(f"ORC")
    ax_bottom.legend(loc='upper left', fontsize=14)

    ax_bottom.set_xlabel(f"Curvature values for LFR (500 nodes, $\mu$={float(args.mus.split(',')[0]):.2f})",
                         fontsize=12)
    ax_top.set_ylabel("Count")
    ax_bottom.set_ylabel("Count")

    ax_top.set_xlim(-2.0,0.8)
    ax_bottom.set_xlim(-2.0,0.8)

    plt.tight_layout()
    plt.savefig(f'./results/result_ricci_curvature_mu{float(args.mus.split(",")[0]):.2f}.pdf')
    plt.close()

if __name__ == "__main__":
    main()
