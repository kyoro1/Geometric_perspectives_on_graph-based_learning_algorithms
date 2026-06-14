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
from src.methods import make_method
from src.curvature import measure_spt_backbone, measure_root_sensitivity_kappa_vs_tree
from src.flow import RicciFlowRunner
from utils import write_dict_rows_to_csv


def run(sbm_nodes: int,
        method: str,
        ratio: float,
        seed: int,
        steps: int,
        delta_or_alpha: float,
        method_kwargs: Dict[str, Any] = None,
        ) -> Dict[str, Any]:

    print('Starting:', method, 'ratio=', ratio)

    ## Initialize data loader
    DL = DataLoader(n=sbm_nodes, seed=seed)
    ## Generate SBM graph
    G0, gt = DL.generate_sbm(pintra=0.15, ratio=ratio)

    ensure_edge_lengths(G0, default_len=1.0, attr="length")

    ## Measure root sensitivity bound-related quantities (LHS vs RHS)
    root_sens_stats = {}
    if method == "src-spt":
        root_sens_stats = measure_root_sensitivity_kappa_vs_tree(
            G0,
            length_attr="length",
            n_roots=20,
            tau=0.8,
            delta=delta_or_alpha,
            p=float(method_kwargs.get("p", 1.0)),
            seed=seed,
        )

    return {
        "method": method,
        "ratio": float(ratio),
        "seed": int(seed),
        "kappa_over_tree_mean": root_sens_stats["kappa_over_tree_mean"],
        "tree_delta_ratio_mean": root_sens_stats["tree_delta_ratio_mean"],
        "kappa_l1_mean": root_sens_stats["kappa_l1_mean"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbm-nodes", type=int, default=500)
    ap.add_argument("--out", type=str, default="fig5_sbm.csv")
    ap.add_argument("--methods", type=str, default="src,orc", help="comma-separated: src,orc")
    ap.add_argument("--ratios", type=str, default="0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument('--sbm_nodes', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ## For SRC params
    ap.add_argument("--src_p", type=float, default=1.0, help="SRC p parameter")
    ap.add_argument("--delta", type=float, default=0.5, help="SRC delta / ORC alpha")
    ap.add_argument("--src_root", type=int, default=0, help="SRC root node (default: 0)")
    ## For ORC params
    ap.add_argument("--orc_calc", type=str, default="equiv", help="ORC calculation method: equiv")
    ap.add_argument("--orc_eta", type=float, default=0.10, help="ORC eta parameter")
    args = ap.parse_args()

    methods = [s.strip().lower() for s in args.methods.split(",") if s.strip()]
    ratios = [float(s.strip()) for s in args.ratios.split(",") if s.strip()]

    tasks = list(product(methods, ratios, range(args.trials)))

    rows: List[Dict[str, Any]] = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(run)(
            method=method,
            ratio=ratio,
            seed=args.seed + trial,
            steps=args.steps,
            delta_or_alpha=args.delta,
            sbm_nodes=args.sbm_nodes,
            method_kwargs={
                "p": args.src_p,
                "src_root": args.src_root,
                "trials": args.trials,
                "orc_calc": args.orc_calc,
                "orc_eta": args.orc_eta,
            },
        )
        for method, ratio, trial in tasks
    )

    # write the results in CSV
    write_dict_rows_to_csv(path=args.out, rows=rows)

if __name__ == "__main__":
    main()
