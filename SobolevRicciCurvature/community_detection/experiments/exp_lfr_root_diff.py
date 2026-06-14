import os
import sys
import time
import argparse
from typing import Dict, Any, List
from joblib import Parallel, delayed
from itertools import product

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets import DataLoader
from src.utils import ensure_edge_lengths
from src.curvature import measure_root_sensitivity_kappa_vs_tree
from utils import write_dict_rows_to_csv

def run(lfr_nodes: int,
        method: str,
        mu: float,
        seed: int,
        steps: int,
        alpha: float,
        method_kwargs: Dict[str, Any] = None,
        ) -> Dict[str, Any]:
    
    ## Initialize data loader
    LFR = DataLoader(n=lfr_nodes, seed=seed)
    G0, gt = LFR.generate_lfr(mu=mu, 
                        avg_deg=20,
                        seed_tries=200,
                        tau1=3.0, tau2=1.5,
                        min_community=20, 
                        max_community=100,
                        max_degree=50)

    ensure_edge_lengths(G0, default_len=1.0, attr="length")

    ## Measure root sensitivity bound-related quantities (LHS vs RHS)
    root_sens_stats = {}
    if method == "src-spt":
        root_sens_stats = measure_root_sensitivity_kappa_vs_tree(
            G0,
            length_attr="length",
            n_roots=20, 
            tau=0.8,
            delta=alpha,
            p=float(method_kwargs.get("p", 1.0)),
            seed=seed,
        )

    return {
        "method": method,
        "mu": float(mu),
        "seed": int(seed),
        "kappa_over_tree_mean": root_sens_stats["kappa_over_tree_mean"],
        "tree_delta_ratio_mean": root_sens_stats["tree_delta_ratio_mean"],
        "kappa_l1_mean": root_sens_stats["kappa_l1_mean"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfr_nodes", type=int, default=500)
    ap.add_argument("--out", type=str, default="fig5_sbm.csv")
    ap.add_argument("--methods", type=str, default="src,orc", help="comma-separated: src,orc")
    ap.add_argument("--mus", type=str, default="0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ## For SRC params
    ap.add_argument("--src_p", type=float, default=1.0, help="SRC p parameter")
    ap.add_argument("--alpha", type=float, default=0.5, help="SRC / ORC alpha")
    ap.add_argument("--src_root", type=int, default=0, help="SRC root node (default: 0)")
    ## For ORC params
    ap.add_argument("--orc_calc", type=str, default="equiv", help="ORC calculation method: equiv")
    ap.add_argument("--orc_eta", type=float, default=0.10, help="ORC eta parameter")
    args = ap.parse_args()

    methods = [s.strip().lower() for s in args.methods.split(",") if s.strip()]
    mus = [float(s.strip()) for s in args.mus.split(",") if s.strip()]

    tasks = list(product(methods, mus, range(args.trials)))

    rows: List[Dict[str, Any]] = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(run)(
            lfr_nodes=args.lfr_nodes,
            method=method,
            mu=mu,
            seed=args.seed + trial,
            steps=args.steps,
            alpha=args.alpha,
            method_kwargs={"p": args.src_p,
                           "orc_calc": args.orc_calc,
                           "orc_eta": args.orc_eta,
                           "trials": args.trials,
                           "t": trial,
                           },
        )
        for method, mu, trial in tasks
    )

    # write the results in CSV
    write_dict_rows_to_csv(path=args.out, rows=rows)


if __name__ == "__main__":
    main()
