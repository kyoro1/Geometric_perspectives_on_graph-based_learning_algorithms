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
from src.flow import RicciFlowRunner
from src.evaluation import best_ari_by_louvain_resolution_k, best_ari_by_length_cut_k, contingency_matrix_general
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

    ## Evaluation via length separation
    print("[cut-eval] start length cut search", flush=True)
    t_cut0 = time.perf_counter()
    best_ari_length, best_thr_length, best_k_length, best_part_length = best_ari_by_length_cut_k(
        G_flow=Gf,
        gt=gt,
        k_target=k_target,
        length_attr="length",
        n_thr=200,
        q_min=0.0,
        q_max=1.0,
        debug=True,
        progress_every=50,
        fallback="closest",
    )
    t_cut1 = time.perf_counter()
    print("[cut-eval] done length cut search", flush=True)

    print(f"[cut-eval][{method}] mu={mu:.3f} seed={seed} ari={best_ari_length:.4f} "
            f"time={st['time_flow_sec']:.2f}s")

    print('[cut-eval] Contingency matrix:')
    contingency_matrix_general(gt=gt, pred=best_part_length)


    ## Evaluation via Louvain
    print("[louvain-eval] start louvain search", flush=True)
    t_luv0 = time.perf_counter()
    best_ari_louvain, best_res_louvain, best_k_louvain, best_part_louvain = best_ari_by_louvain_resolution_k(
        Gf, gt,
        k_target=k_target,
        length_attr="length",
        beta=1.0,
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
    print("[debug] pred_length size:", len(best_part_length), "unique pred:", len(set(best_part_length.values())))
    print("[debug] pred_louvain size:", len(best_part_louvain), "unique pred:", len(set(best_part_louvain.values())))

    common = set(gt.keys()) & set(best_part_louvain.keys())
    print("[debug] common nodes(gt ∩ pred_louvain):", len(common))

    return {
        "method": method,
        "mu": float(mu),
        "seed": int(seed),
        "alpha": float(alpha),
        "steps": int(steps),
        ## Flow results
        "time_flow_sec": float(st["time_flow_sec"]),
        ## Louvain results
        "best_ari_louvain": float(best_ari_louvain),
        "best_res_louvain": float(best_res_louvain),
        "k_pred_louvain": int(best_k_louvain),
        "time_louvain_sec": float(t_luv1 - t_luv0),
        ## Length cut results
        "best_ari_length": float(best_ari_length),
        "best_thr_length": float(best_thr_length),
        "k_pred_length": int(best_k_length),
        "time_cut_sec_for_length": float(t_cut1 - t_cut0),
        ## Basic graph info
        "number_of_nodes": int(G0.number_of_nodes()),
        "number_of_edges": int(G0.number_of_edges()),
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
    ## For SRC param
    ap.add_argument("--src_p", type=str, default=1.0, help="SRC p parameter")
    ap.add_argument("--alpha", type=str, default="0.5", help="alpha")
    ## For ORC params
    ap.add_argument("--orc_calc", type=str, default="equiv", help="ORC calculation method: equiv")
    ap.add_argument("--orc_eta", type=float, default=0.10, help="ORC eta parameter")
    args = ap.parse_args()

    methods = [s.strip().lower() for s in args.methods.split(",") if s.strip()]
    mus = [float(s.strip()) for s in args.mus.split(",") if s.strip()]
    alphas = [float(s.strip()) for s in args.alpha.split(",") if s.strip()]

    tasks = list(product(methods, mus, alphas, range(args.trials)))

    rows: List[Dict[str, Any]] = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(run)(
            lfr_nodes=args.lfr_nodes,
            method=method,
            mu=mu,
            seed=args.seed + trial,
            steps=args.steps,
            alpha=alpha,
            method_kwargs={"p": args.src_p,
                           "orc_calc": args.orc_calc,
                           "orc_eta": args.orc_eta,
                           "trials": args.trials,
                           "t": trial,
                           },
        )
        for method, mu, alpha, trial in tasks
    )

    # write the results in CSV
    write_dict_rows_to_csv(path=args.out, rows=rows)

if __name__ == "__main__":
    main()
