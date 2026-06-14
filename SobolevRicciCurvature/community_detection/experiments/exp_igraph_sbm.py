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
from src.community import partition_by_igraph
from src.evaluation import calc_ari_with_labels
from utils import write_dict_rows_to_csv

def run(sbm_nodes: int,
        method: str,
        ratio: float,
        seed: int,
        ) -> Dict[str, Any]:
    ## Preparation
    start = time.time()
    print('Starting:', method, 'ratio=', ratio)
    ## Initialize data loader
    DL = DataLoader(n=sbm_nodes, seed=seed)
    ## Generate SBM graph
    G0, gt = DL.generate_sbm(pintra=0.15, ratio=ratio)

    ## Community detection via igraph
    labels = partition_by_igraph(G=G0, method=method)
    
    ## evaluate
    ari = calc_ari_with_labels(labels, gt)
    time_elapsed = time.time() - start
    print(f"Method: {method}, Ratio: {ratio}, ARI: {ari:.4f}, Time: {time_elapsed:.2f}s")

    return {
        "method": method,
        "ratio": ratio,
        "ari": ari,
        "elapsed": time_elapsed,
        ## Basic graph info
        "number_of_nodes": int(G0.number_of_nodes()),
        "number_of_edges": int(G0.number_of_edges()),
        "directed": bool(G0.is_directed()),
        "max_LCC_size": int(DL.calc_max_lcc_size(G0)),    
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbm_nodes", type=int, default=500)
    ap.add_argument("--methods", type=str, default='infomap', help="community detection method to use (infomap, spinglass, fastgreedy)")
    ap.add_argument("--ratios", type=str, default="0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="exp_igraph_results.csv")
    args = ap.parse_args()

    methods = [s.strip().lower() for s in args.methods.split(",") if s.strip()]
    ratios = [float(s.strip()) for s in args.ratios.split(",") if s.strip()]

    tasks = product(methods, ratios, range(args.trials))

    rows: List[Dict[str, Any]] = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(run)(
            sbm_nodes=args.sbm_nodes,
            method=method,
            ratio=ratio,
            seed=args.seed + trial,
        )
        for method, ratio, trial in tasks
    )
    
    # write the results in CSV
    write_dict_rows_to_csv(path=args.out, rows=rows)

if __name__ == "__main__":
    main()    