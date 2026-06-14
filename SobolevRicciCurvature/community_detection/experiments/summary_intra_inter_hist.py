import os
import sys
import argparse
import networkx as nx
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets import DataLoader
from src.utils import ensure_edge_lengths
from src.methods import make_method
from src.flow import RicciFlowRunner

import numpy as np
import matplotlib.pyplot as plt

def extract_src_edge_curvatures(st: Dict[str, Any], mode: str = "final") -> Dict[tuple, float]:
    K = st["kappa_hist"]      # (T, E)
    u_idx = st["u_idx"]
    v_idx = st["v_idx"]
    nodes = np.array(st["nodes"], dtype=np.int32)

    if mode == "final":
        kappa_e = K[-1]
    elif mode == "min":
        kappa_e = K.min(axis=0)
    elif mode == "avg_last":
        kappa_e = K[-10:].mean(axis=0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    kappa_dict = {}
    for uu, vv, kk in zip(u_idx, v_idx, kappa_e):
        a = int(nodes[int(uu)])
        b = int(nodes[int(vv)])
        if a > b:
            a, b = b, a
        kappa_dict[(a, b)] = float(kk)
    return kappa_dict


def extract_orc_edge_curvatures(Gf: nx.Graph) -> Dict[tuple, float]:
    rc = nx.get_edge_attributes(Gf, "ricciCurvature")
    kappa_dict = {}
    for (a, b), val in rc.items():
        if a > b:
            a, b = b, a
        kappa_dict[(a, b)] = float(val)
    return kappa_dict


# -------------------------
# main runner
# -------------------------
def run_and_collect(
    lfr_nodes: int,
    method: str,
    mu: float,
    seed: int,
    steps: int,
    delta_or_alpha: float,
    method_kwargs: Dict[str, Any] = None,
    src_mode: str = "final",
):
    print("Starting:", method, "mu=", mu)

    LFR = DataLoader(n=lfr_nodes, seed=seed)
    G0, gt_dict = LFR.generate_lfr(
        mu=mu,
        avg_deg=20,
        seed_tries=200,
        tau1=3.0,
        tau2=1.5,
        min_community=20,
        max_community=100,
        max_degree=50,
    )
    ensure_edge_lengths(G0, default_len=1.0, attr="length")

    method_obj = make_method(method, steps, delta_or_alpha, method_kwargs=method_kwargs)
    runner = RicciFlowRunner(method_obj)
    Gf, st = runner.run(G0, seed=seed)

    # curvature dict
    if method in ["src-spt", "src-mst"]:
        kappa_dict = extract_src_edge_curvatures(st, mode=src_mode)
    else:
        kappa_dict = extract_orc_edge_curvatures(Gf)

    edges = sorted(list(kappa_dict.keys()))
    kappa_vals = np.array([kappa_dict[e] for e in edges], dtype=np.float64)
    inter_mask = np.array([gt_dict[a] != gt_dict[b] for (a, b) in edges], dtype=bool)

    print(f"[sanity] {method:7s} m={len(edges):5d} inter={inter_mask.mean():.3f} "
          f"kappa[min,max]=({kappa_vals.min():.3f},{kappa_vals.max():.3f})")

    return kappa_vals, inter_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfr_nodes", type=int, default=500)
    ap.add_argument("--methods", type=str, default="orc,src-spt,src-mst")
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src_p", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=0.5)
    ap.add_argument("--src_mode", type=str, default="final", choices=["final", "min", "avg_last"])
    ap.add_argument("--bins", type=int, default=120)
    ap.add_argument("--out", type=str, default="./results/result_intra_inter_hist.pdf")
    args = ap.parse_args()

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    color_map = {"orc": "C0", "src-mst": "C1", "src-spt": "C2"}

    # ---- collect ----
    data = {}
    all_vals = []

    for method in methods:
        kappa_vals, inter_mask = run_and_collect(
            lfr_nodes=args.lfr_nodes,
            method=method,
            mu=args.mu,
            seed=args.seed,
            steps=args.steps,
            delta_or_alpha=args.delta,
            method_kwargs={"p": args.src_p},
            src_mode=args.src_mode,
        )
        data[method] = (kappa_vals, inter_mask)
        all_vals.append(kappa_vals)

    all_vals = np.concatenate(all_vals)
    bin_edges = np.histogram_bin_edges(all_vals, bins=args.bins)

    # ---- plot ----
    fig, axes = plt.subplots(
        len(methods), 1,
        figsize=(7.5, 2.4 * len(methods)),
        sharex=True
    )
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        kappa_vals, inter_mask = data[method]
        intra_vals = kappa_vals[~inter_mask]
        inter_vals = kappa_vals[inter_mask]

        # intra: filled (light)
        ax.hist(
            intra_vals,
            bins=bin_edges,
            alpha=0.35,
            histtype="stepfilled",
            color=color_map.get(method, None),
            label="intra",
        )
        # inter: outline (strong)
        ax.hist(
            inter_vals,
            bins=bin_edges,
            alpha=0.95,
            histtype="step",
            linewidth=1.8,
            color=color_map.get(method, None),
            label="inter",
        )

        ax.set_title(method, fontsize=12)
        ax.set_ylabel("Count")
        ax.legend(loc="upper left")

    axes[-1].set_xlabel(f"Curvature values (LFR n={args.lfr_nodes}, μ={args.mu:.2f})")

    fig.suptitle("Intra vs Inter curvature distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out)
    plt.close()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
