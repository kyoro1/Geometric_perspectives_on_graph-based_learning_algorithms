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
from src.community import partition_by_igraph
from src.evaluation import calc_ari_with_labels
from utils import write_dict_rows_to_csv

def run(method: str,
        name: str,
        seed: int,
        ) -> Dict[str, Any]:
    ## Preparation
    start = time.time()
    print('Starting:', method, 'name=', name, 'seed=', seed)

    ## Initialize data loader
    RD = DataLoader(n=None, seed=seed)
    G0, gt = RD.load_real_graph(name=name, data_dir='./data/')

    ## Community detection via igraph
    labels = partition_by_igraph(G=G0, method=method, seed=seed)

    print('len(labels) =', len(labels), 'len(gt)=', len(gt), "len(set(gt.values())) =", len(set(gt.values())), 'len(set(gt.keys()) & set(range(len(labels))))=', len(set(gt.keys()) & set(range(len(labels)))) )

    print("labels type:", type(labels))
    print("gt type:", type(gt))
    print("labels len:", len(labels) if hasattr(labels,'__len__') else None)
    print("gt len:", len(gt) if gt is not None else None)

    if isinstance(labels, np.ndarray):
        pred_keys = set(range(len(labels)))
    else:
        pred_keys = set(labels.keys())

    gt_keys = set(gt.keys())
    print("intersection:", len(pred_keys & gt_keys))
    print("gt sample keys:", list(sorted(gt_keys))[:10])
    print("pred sample keys:", list(sorted(pred_keys))[:10])


    ## evaluate
    ari = calc_ari_with_labels(labels, gt)
    time_elapsed = time.time() - start
    print(f"Method: {method}, ARI: {ari:.4f}, Time: {time_elapsed:.2f}s")

    return {
        "method": method,
        "name": name,
        "ari": ari,
        "elapsed": time_elapsed,
        "seed": seed,
        ## Basic graph info
        "number_of_nodes": int(G0.number_of_nodes()),
        "number_of_edges": int(G0.number_of_edges()),
        "directed": bool(G0.is_directed()),
        "max_LCC_size": int(RD.calc_max_lcc_size(G0)),    
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", type=str, default='infomap', help="community detection method to use (infomap, spinglass, fastgreedy)")
    ap.add_argument("--names", type=str, default="karate")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="exp_igraph_real_results.csv")
    args = ap.parse_args()

    methods = [s.strip().lower() for s in args.methods.split(",") if s.strip()]
    names = [s.strip() for s in args.names.split(",") if s.strip()]

    rows: List[Dict[str, Any]] = []
    for method in methods:
        for name in names:
            print(f"Running method: {method}, dataset: {name}")

            if method == "spinglass" or method == "label_propagation":
                print(f"[Warning] method {method} is stochastic, running {args.trials} trials with different seeds.")
                for trial in range(args.trials):
                    print(args.seed + trial)
                    r = run(
                        method=method,
                        name=name,
                        seed=args.seed + trial,
                    )
                    rows.append(r)
            else:
                r = run(
                    method=method,
                    name=name,
                    seed=args.seed,
                )
                rows.append(r)

    # write the results in CSV
    write_dict_rows_to_csv(path=args.out, rows=rows)

if __name__ == "__main__":
    main()    