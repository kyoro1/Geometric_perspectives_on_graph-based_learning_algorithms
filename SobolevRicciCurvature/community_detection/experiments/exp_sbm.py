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
from src.evaluation import best_ari_by_louvain_resolution_k, contingency_matrix, best_ari_by_length_cut_k
from utils import write_dict_rows_to_csv


def run(sbm_nodes: int,
        method: str,
        ratio: float,
        seed: int,
        steps: int,
        alpha: float,
        method_kwargs: Dict[str, Any] = None,
        ) -> Dict[str, Any]:

    ## Preparation
    start = time.time()
    print('Starting:', method, 'ratio=', ratio)

    ## Initialize data loader
    DL = DataLoader(n=sbm_nodes, seed=seed)
    ## Generate SBM graph
    G0, gt = DL.generate_sbm(pintra=0.15, ratio=ratio)


    ensure_edge_lengths(G0, default_len=1.0, attr="length")

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
        k_target=2,
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

    print(f"[cut-eval][{method}] ratio={ratio:.3f}, ari={best_ari_length:.4f} "
            f"time={st['time_flow_sec']:.2f}s")

    print('[cut-eval] Contingency matrix:')
    contingency_matrix(gt=gt, pred=best_part_length)

    ## Evaluation via Louvain
    print("[louvain-eval] start louvain search", flush=True)
    t_luv0 = time.perf_counter()
    best_ari_louvain, best_res_louvain, best_k_louvain, best_part_louvain = best_ari_by_louvain_resolution_k(
        Gf, gt,
        k_target=2,
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

    print(f"[louvain-eval][{method}] ratio={ratio:.3f}, ari={best_ari_louvain:.4f} "
            f"time={st['time_flow_sec']:.2f}s")

    process_time = time.time() - start
    print('[louvain-eval] Finished:', method, 'ratio=', ratio, 'time=', process_time)

    ## display contingency matrix
    print('[louvain-eval] Contingency matrix:')
    contingency_matrix(gt=gt, pred=best_part_louvain)

    return {
        "method": method,
        "ratio": float(ratio),
        "seed": int(seed),
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
        "max_LCC_size": int(DL.calc_max_lcc_size(G0)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbm-nodes", type=int, default=500)
    ap.add_argument("--out", type=str, default="fig5_sbm.csv")
    ap.add_argument("--methods", type=str, default="src,orc", help="comma-separated: src,orc")
    ap.add_argument("--ratios", type=str, default="0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ## For SRC params
    ap.add_argument("--src_p", type=float, default=1.0, help="SRC p parameter")
    ap.add_argument("--alpha", type=float, default=0.5, help="SRC alpha / ORC alpha")
    ap.add_argument("--src_root", type=int, default=0, help="SRC root node (default: 0)")
    ## For ORC params
    ap.add_argument("--orc_calc", type=str, default="equiv", help="ORC calculation method: equiv")
    ap.add_argument("--orc_eta", type=float, default=0.10, help="ORC eta parameter")
    args = ap.parse_args()

    methods = [s.strip().lower() for s in args.methods.split(",") if s.strip()]
    ratios = [float(s.strip()) for s in args.ratios.split(",") if s.strip()]
    tasks = list(product(methods, ratios, range(args.trials)))

    rows: List[Dict[str, Any]] = Parallel(n_jobs=-1, backend="loky")(
        delayed(run)(
            method=method,
            ratio=ratio,
            seed=args.seed + trial,
            steps=args.steps,
            alpha=args.alpha,
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
