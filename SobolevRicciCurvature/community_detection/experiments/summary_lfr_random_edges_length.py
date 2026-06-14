import os, sys
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets import DataLoader
from src.flow import SRCFlowMethod


def plot_hist_three_rows(
    kappa_spt, kappa_mst, kappa_rand,
    mu: float,
    n_nodes: int,
    bins: int = 150,
    xlim=(-2.0, 0.8),
    density: bool = False,
    alpha: float = 0.55,
    out_path: str = "lfr_random_edges_length_curvature_hist.pdf",
):
    def clean(x):
        x = np.asarray(x).ravel()
        return x[np.isfinite(x)]

    k_spt = clean(kappa_spt)
    k_mst = clean(kappa_mst)
    k_rnd = clean(kappa_rand)

    left, right = xlim
    common_bins = np.linspace(left, right, bins + 1)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.2), sharex=True, constrained_layout=True)

    ylabel = "Density" if density else "Count"

    c_spt, _ = np.histogram(k_spt, bins=common_bins, density=density)
    c_mst, _ = np.histogram(k_mst, bins=common_bins, density=density)
    c_rnd, _ = np.histogram(k_rnd, bins=common_bins, density=density)
    y_max = float(np.max([c_spt.max(initial=0), c_mst.max(initial=0), c_rnd.max(initial=0)]))
    y_max *= 1.05 if y_max > 0 else 1.0

    title_fontsize = 13

    # --- 1) SRC-SPT ---
    axes[0].hist(k_spt, bins=common_bins, alpha=alpha, density=density, color="C2")
    axes[0].set_title("SRC(SPT)", fontsize=title_fontsize, pad=2)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(0, y_max)

    # --- 2) SRC-MST ---
    axes[1].hist(k_mst, bins=common_bins, alpha=alpha, density=density, color="C1")
    axes[1].set_title("SRC(MST)", fontsize=title_fontsize, pad=2)
    axes[1].set_ylabel(ylabel)
    axes[1].set_ylim(0, y_max)

    # --- 3) SRC-Random ---
    axes[2].hist(k_rnd, bins=common_bins, alpha=alpha, density=density, color="magenta")
    axes[2].set_title("SRC-Random", fontsize=title_fontsize, pad=2)
    axes[2].set_ylabel(ylabel)
    axes[2].set_ylim(0, y_max)
    axes[2].set_xlabel(
        f"Curvature values for LFR ({n_nodes} nodes, p=0.10, mu={mu})",
        labelpad=2,
    )
    axes[2].set_xlim(left, right)

    grid_kwargs = dict(
        which="both",
        axis="both",
        linestyle="--",
        linewidth=0.6,
        color="0.85",
        alpha=0.8,
    )
    for ax in axes:
        ax.grid(True, **grid_kwargs)
        ax.set_axisbelow(True)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)

def assign_random_edge_lengths(G: nx.Graph, length_attr: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G[u][v][length_attr] = float(rng.uniform(0.0, 1.0))


def run_once(G0: nx.Graph, method: str, delta: float, steps: int, p: float, src_root: int):
    flow = SRCFlowMethod(delta=delta, steps=steps, p=p, method=method, src_root=src_root)
    _, st = flow.run(G0)
    return st["kappa_hist"][0], st


def build_argparser():
    ap = argparse.ArgumentParser(description="LFR + random edge lengths: SRC(SPT/MST/RandTree) histogram plot")

    # LFR
    ap.add_argument("--n", type=int, default=500, help="number of nodes")
    ap.add_argument("--seed", type=int, default=42, help="seed for LFR generation")
    ap.add_argument("--mu", type=float, default=0.4, help="LFR mixing parameter mu")
    ap.add_argument("--avg-deg", type=int, default=20, help="LFR avg degree")
    ap.add_argument("--seed-tries", type=int, default=200, help="LFR seed tries")
    ap.add_argument("--tau1", type=float, default=3.0, help="LFR tau1")
    ap.add_argument("--tau2", type=float, default=1.5, help="LFR tau2")
    ap.add_argument("--min-community", type=int, default=20, help="LFR min community size")
    ap.add_argument("--max-community", type=int, default=100, help="LFR max community size")
    ap.add_argument("--max-degree", type=int, default=50, help="LFR max degree")

    # edge lengths
    ap.add_argument("--length-attr", type=str, default="length", help="edge attribute name for length")
    ap.add_argument("--length-seed", type=int, default=0, help="seed for Uniform[0,1] edge lengths")

    # SRC params
    ap.add_argument("--delta", type=float, default=0.5, help="SRC delta")
    ap.add_argument("--p", type=float, default=1.0, help="SRC p (W_p)")
    ap.add_argument("--steps", type=int, default=1, help="SRC flow steps (NOTE: steps=1 still performs one update inside run())")
    ap.add_argument("--src-root", type=int, default=0, help="root node for tree construction")

    # plot params
    ap.add_argument("--bins", type=int, default=150, help="hist bins")
    ap.add_argument("--xlim-left", type=float, default=-2.0, help="x-axis left limit")
    ap.add_argument("--xlim-right", type=float, default=0.8, help="x-axis right limit")
    ap.add_argument("--density", action="store_true", help="plot density instead of count")
    ap.add_argument("--out", type=str, default="lfr_random_edges_length_curvature_hist.pdf", help="output file path")

    return ap


def main():
    args = build_argparser().parse_args()

    # 1) LFR graph
    LFR = DataLoader(n=args.n, seed=args.seed)
    G0, gt = LFR.generate_lfr(
        mu=args.mu,
        avg_deg=args.avg_deg,
        seed_tries=args.seed_tries,
        tau1=args.tau1, tau2=args.tau2,
        min_community=args.min_community,
        max_community=args.max_community,
        max_degree=args.max_degree,
    )

    # 2) random edge lengths Uniform[0,1]
    assign_random_edge_lengths(G0, length_attr=args.length_attr, seed=args.length_seed)

    # 3) compute kappas
    k_spt, _ = run_once(G0, "src-spt", delta=args.delta, steps=args.steps, p=args.p, src_root=args.src_root)
    k_mst, _ = run_once(G0, "src-mst", delta=args.delta, steps=args.steps, p=args.p, src_root=args.src_root)
    k_rnd, _ = run_once(G0, "src-randtree", delta=args.delta, steps=args.steps, p=args.p, src_root=args.src_root)

    # 4) plot (3x1)
    plot_hist_three_rows(
        kappa_spt=k_spt,
        kappa_mst=k_mst,
        kappa_rand=k_rnd,
        mu=args.mu,
        n_nodes=args.n,
        bins=args.bins,
        xlim=(args.xlim_left, args.xlim_right),
        density=args.density,
        out_path=args.out,
        alpha=0.5,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
