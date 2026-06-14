import os
import sys
import time
import argparse
from typing import Dict, Any
from joblib import Parallel, delayed
import numpy as np
import networkx as nx

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets import DataLoader
from src.utils import ensure_edge_lengths
from src.methods import make_method
from src.flow import RicciFlowRunner
from src.evaluation import best_ari_by_louvain_resolution_k, best_ari_by_length_cut_k, contingency_matrix_general
from utils import write_dict_rows_to_csv

def run(method: str,
        name: str,
        seed: int,
        steps: int,
        alpha: float,
        method_kwargs: Dict[str, Any] = None,
        ) -> Dict[str, Any]:

    ## Preparation
    start = time.time()
    print('Starting:', method)
    ## Initialize data loader
    RD = DataLoader(n=None, seed=seed)
    G0, gt = RD.load_real_graph(name=name, data_dir='./data/')

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

    print(f"[cut-eval][{method}] name={name},ari={best_ari_length:.4f} "
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
        beta=0.5,
        res_min=0.2,
        res_max=4.0,
        n_res=60,
        seed=seed,
        debug=True,
    )
    t_luv1 = time.perf_counter()
    print("[louvain-eval] Done louvain search", flush=True)

    print(f"[louvain-eval][{method}] name={name}, ari={best_ari_louvain:.4f} "
            f"time={st['time_flow_sec']:.2f}s")

    process_time = time.time() - start
    print('[louvain-eval] Finished:', method, 'name=', name, 'time=', process_time)
    ## display contingency matrix
    print('[louvain-eval] Contingency matrix:')
    contingency_matrix_general(gt=gt, pred=best_part_louvain)

    return {
        "method": method,
        "name": name,
        "seed": int(seed),
        "alpha": float(alpha),
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
        "max_LCC_size": int(RD.calc_max_lcc_size(G0)),
        "k_gt": k_target,
        }

def set_root_nodes(name: str, size: int) -> int:
    ## set number of nodes
    if name == 'karate':
        nodes = 34
    elif name == 'football' or name == 'polbooks':
        nodes = 100
    else:
        nodes = 1000
    print("[debug] set_root_nodes:", name, "nodes:", nodes, "size:", size)
    ## return random root nodes
    return np.random.randint(0, nodes, size=size)

def _run_one(method, name, seed, steps, alpha, method_kwargs):
    print(f"[task] method={method} name={name} seed={seed} steps={steps} alpha={alpha} kwargs={method_kwargs}")
    return run(
        method=method,
        name=name,
        seed=seed,
        steps=steps,
        alpha=alpha,
        method_kwargs=method_kwargs,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=str, default="karate")
    ap.add_argument("--out", type=str, default="lfr_src.csv")
    ap.add_argument("--methods", type=str, default="src,orc", help="src,orc")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ## For SRC params
    ap.add_argument("--src_p", type=float, default=1.0, help="SRC p parameter")
    ap.add_argument("--alphas", type=str, default="0.5", help="SRC / ORC alpha values, comma-separated")
    ## For ORC params
    ap.add_argument("--orc_calc", type=str, default="equiv", help="ORC calculation method: equiv")
    ap.add_argument("--orc_eta", type=float, default=0.10, help="ORC eta parameter")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    methods = [s.strip().lower() for s in args.methods.split(",") if s.strip()]
    names = [s.strip() for s in args.names.split(",") if s.strip()]
    alphas = [float(s.strip()) for s in args.alphas.split(",") if s.strip()]

    tasks = []
    task_id = 0

    for method in methods:
        for name in names:
            for alpha in alphas:
                if method == "src-spt":
                    src_rootnodes = set_root_nodes(name=name, size=args.trials)

                    for src_root in src_rootnodes:
                        method_kwargs = {
                            "p": args.src_p,
                            "trials": args.trials,
                            "orc_calc": args.orc_calc,
                            "orc_eta": args.orc_eta,
                            "src_root": src_root,
                        }
                        seed_i = args.seed + task_id
                        tasks.append((method, name, seed_i, args.steps, alpha, method_kwargs))
                        task_id += 1
                else:
                    method_kwargs = {
                        "p": args.src_p,
                        "trials": args.trials,
                        "orc_calc": args.orc_calc,
                        "orc_eta": args.orc_eta,
                    }
                    seed_i = args.seed + task_id
                    tasks.append((method, name, seed_i, args.steps, alpha, method_kwargs))
                    task_id += 1

    rows = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(_run_one)(*t) for t in tasks
    )
    # write the results in CSV
    write_dict_rows_to_csv(path=args.out, rows=rows)

if __name__ == "__main__":
    main()
