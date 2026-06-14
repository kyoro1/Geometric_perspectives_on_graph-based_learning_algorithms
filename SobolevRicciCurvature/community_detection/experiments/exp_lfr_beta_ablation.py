import os
import sys
import time
import argparse
from typing import Dict, Any, List
from joblib import Parallel, delayed
from itertools import product
import numpy as np

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets import DataLoader
from src.utils import ensure_edge_lengths
from src.methods import make_method
from src.flow import RicciFlowRunner
from src.evaluation import best_ari_by_louvain_resolution_k, contingency_matrix_general
from utils import write_dict_rows_to_csv

def run(lfr_nodes: int,
        method: str,
        mu: float,
        seed: int,
        steps: int,
        alpha: float,
        method_kwargs: Dict[str, Any] = None,
        ) -> Dict[str, Any]:

    ## Preparation
    start = time.time()
    print('Starting:', method, 'mu=', mu)

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
    k_target = len(set(gt.values()))

    ## Select method: 'src' or 'orc'
    method_obj = make_method(method, 
                             steps, 
                             alpha, 
                             method_kwargs=method_kwargs,
                             )

    runner = RicciFlowRunner(method_obj)

    Gf, st = runner.run(G0, seed=seed)

    ## Evaluation via Louvain
    print("[louvain-eval] start louvain search", flush=True)
    t_luv0 = time.perf_counter()
    best_ari_louvain, best_res_louvain, best_k_louvain, best_part_louvain = best_ari_by_louvain_resolution_k(
        Gf, gt,
        k_target=k_target,
        length_attr="length",
        beta=method_kwargs['beta'] if method_kwargs and 'beta' in method_kwargs else 0.1,  
        res_min=0.2,
        res_max=4.0,
        n_res=60,
        seed=seed,
        debug=True,
    )
    t_luv1 = time.perf_counter()
    print("[louvain-eval] Done louvain search", flush=True)

    print(f"[louvain-eval][{method}] mu={mu:.3f}, ari={best_ari_louvain:.4f} "
            f"time={st['time_flow_sec']:.2f}s")

    process_time = time.time() - start
    print('[louvain-eval] Finished:', method, 'mu=', mu, 'seed=', seed, 'time=', process_time)
    ## display contingency matrix
    print('[louvain-eval] Contingency matrix:')
    contingency_matrix_general(gt=gt, pred=best_part_louvain)

    print("[debug] G0 nodes/edges:", G0.number_of_nodes(), G0.number_of_edges())
    print("[debug] Gf nodes/edges:", Gf.number_of_nodes(), Gf.number_of_edges())

    print("[debug] gt size:", len(gt), "unique gt labels:", len(set(gt.values())))
    print("[debug] pred_louvain size:", len(best_part_louvain), "unique pred:", len(set(best_part_louvain.values())))

    common = set(gt.keys()) & set(best_part_louvain.keys())
    print("[debug] common nodes(gt ∩ pred_louvain):", len(common))

    return {
        "method": method,
        "mu": float(mu),
        "seed": int(seed),
        ## Flow results
        "time_flow_sec": float(st["time_flow_sec"]),
        ## Louvain results
        "best_ari_louvain": float(best_ari_louvain),
        "best_res_louvain": float(best_res_louvain),
        "k_pred_louvain": int(best_k_louvain),
        "time_louvain_sec": float(t_luv1 - t_luv0),
        ## Basic graph info
        "number_of_nodes": int(G0.number_of_nodes()),
        "number_of_edges": int(G0.number_of_edges()),
        "src_p": float(method_kwargs["p"]) if method_kwargs and "p" in method_kwargs else None,
        "beta": float(method_kwargs["beta"]) if method_kwargs and "beta" in method_kwargs else None,
        "directed": bool(G0.is_directed()),
        "max_LCC_size": int(LFR.calc_max_lcc_size(G0)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfr_nodes", type=int, default=500)
    ap.add_argument("--out", type=str, default="lfr_src.csv")
    ap.add_argument("--methods", type=str, default="src-spt,src-mst,orc", help="comma-separated: src-spt,src-mst,orc")
    ap.add_argument("--mus", type=str, default="0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ## For SRC params
    ap.add_argument("--src_p", type=str, default="1.0,1.5,2.0", help="SRC p parameter")
    ap.add_argument("--alpha", type=float, default=0.5, help="SRC alpha / ORC alpha")
    ap.add_argument("--betas", type=str, default="0.1", help="SRC beta parameter")
    args = ap.parse_args()

    methods = [s.strip().lower() for s in args.methods.split(",") if s.strip()]
    mus = [float(s.strip()) for s in args.mus.split(",") if s.strip()]
    src_ps = [float(s.strip()) for s in args.src_p.split(",") if s.strip()]
    betas = [float(s.strip()) for s in args.betas.split(",") if s.strip()]

    tasks = list(product(methods, mus, src_ps, betas, range(args.trials)))

    rows: List[Dict[str, Any]] = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(run)(
            lfr_nodes=args.lfr_nodes,
            method=method,
            mu=mu,
            seed=args.seed + trial,
            steps=args.steps,
            alpha=args.alpha,
            method_kwargs={"p": src_p,
                           "trials": args.trials,
                           "t": trial,
                           "beta": beta,
                           },
        )
        for method, mu, src_p, beta, trial in tasks
    )

    # write the results in CSV
    write_dict_rows_to_csv(path=args.out, rows=rows)

if __name__ == "__main__":
    main()
