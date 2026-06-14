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
from src.community import partition_by_igraph
from src.evaluation import calc_ari_with_labels
from utils import write_dict_rows_to_csv


def run(lfr_nodes: int,
        method: str,
        mu: float,
        seed: int,
        ) -> Dict[str, Any]:
    ## Preparation
    start = time.time()
    print('Starting:', method, 'mu=', mu, 'seed=', seed)

    ## Initialize data loader
    LFR = DataLoader(n=lfr_nodes, seed=seed)
    G0, gt = LFR.generate_lfr(mu=mu, 
                        avg_deg=20,
                        seed_tries=200,
                        tau1=3.0, tau2=1.5,
                        min_community=20, 
                        max_community=100,
                        max_degree=50)    ## Generate SBM graph

    ## Community detection via igraph
    labels = partition_by_igraph(G=G0, 
                               method=method)
    
    ## evaluate
    ari = calc_ari_with_labels(labels, gt)
    time_elapsed = time.time() - start
    print(f"Method: {method}, Mu: {mu}, Seed: {seed}, ARI: {ari:.4f}, Time: {time_elapsed:.2f}s")

    return {
        "method": method,
        "mu": mu,
        "ari": ari,
        "elapsed": time_elapsed,
        ## Basic graph info
        "number_of_nodes": int(G0.number_of_nodes()),
        "number_of_edges": int(G0.number_of_edges()),
        "directed": bool(G0.is_directed()),
        "max_LCC_size": int(LFR.calc_max_lcc_size(G0)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfr_nodes", type=int, default=500)
    ap.add_argument("--methods", type=str, default='infomap', help="community detection method to use (infomap, spinglass, fastgreedy)")
    ap.add_argument("--mus", type=str, default="0.1,0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="exp_igraph_results.csv")
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
        )
        for method, mu, trial in tasks
    )

    # write the results in CSV
    write_dict_rows_to_csv(path=args.out, rows=rows)

if __name__ == "__main__":
    main()    