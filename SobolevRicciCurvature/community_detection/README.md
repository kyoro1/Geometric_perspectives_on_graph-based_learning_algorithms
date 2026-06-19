# Community Detection Experiments with SRC/ORC

This subdirectory contains the community-detection implementation and reproducibility scripts used in the paper.

The core method is **Sobolev-Ricci Flow (SRF / SRC)**, evaluated against **Ollivier–Ricci Curvature (ORC)** and iGraph baselines.
In practice, experiments are organized into:

- benchmark runners on SBM/LFR/real datasets (for example, `experiments/exp_sbm.py`, `experiments/exp_lfr.py`, `experiments/exp_real.py`)
- summary scripts that aggregate ARI/process-time and ablation results (for example, `experiments/summary_sbm.py`, `experiments/summary_lfr.py`, `experiments/summary_real.py`)
- ORC/SRC gap analysis and visualization for cycle-controlled graphs (for example, `experiments/evaluate_orc_src_gap_with_edge_selection_counts.py`, `experiments/plot_orc_src_gap_compare_methods_with_beta1.py`)

The ORC-related components build upon [GraphRicciCurvature](https://github.com/saibalmars/GraphRicciCurvature), which is vendored in this subdirectory under [./src/GraphRicciCurvature](./src/GraphRicciCurvature).

---

# Reproductibility for the dissertation
## Setup

Please set up `venv` setup as follows:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ./requirements.txt
```

## Implementation
These scripts focus on **community-detection strategies** motivated by curvatures. Each script corrsponds to figures on the paper.

## For SBM on `Figure 1(a), (d)`
### SRC & ORC algorithms

```sh
nohup python -u experiments/exp_sbm.py \
  --method src-spt,src-mst,orc \
  --src_p 1.0 \
  --sbm-nodes 500 \
  --steps 50 \
  --ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --trials 10 \
  --src_root 0 \
  --out ./results/sbm_src_orc.csv > logs/sbm_src_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Other algorithms with `iGraph`

```sh
nohup python -u ./experiments/exp_igraph_sbm.py \
  --sbm_nodes 500 \
  --methods infomap,spinglass,fastgreedy,edgebetweenness,label_propagation \
  --ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --trials 10 \
  --out ./results/sbm_igraph.csv > logs/result_igraph_sbm_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Summary script (for `Figure 1(a)`)

```sh
python experiments/summary_sbm.py
```
### Summary report on process time (for `Figure 1(d)`)

```sh
python ./experiments/summary_sbm_process_time.py
```

## LFR on `Figure 1(b)`
### SRC & ORC

```sh
nohup python -u experiments/exp_lfr.py \
  --lfr_nodes 500 \
  --method src-spt,src-mst,orc \
  --src_p 1.0 \
  --steps 50 \
  --mus 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --trials 10 \
  --out ./results/lfr_orc_src.csv > logs/lfr_src_orc_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### (optional) Process time for LFR dataset

```sh
python experiments/summary_lfr_process_time.py
```


### iGraph

```sh
nohup python -u experiments/exp_lfr_igraph.py \
  --lfr_nodes 500 \
  --mus 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --methods infomap,spinglass,fastgreedy,edgebetweenness,label_propagation \
  --trials 10 \
  --out ./results/lfr_igraph.csv > logs/lfr_igraph_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Summary script on `Figure 1(b)`

```sh
python ./experiments/summary_lfr.py
```

## Real datasets on `Figure 1(c)`

### SRC & ORC algorithms
```sh
nohup python -u experiments/exp_real.py \
  --name karate,football,polbooks,polblogs,email-eu-core \
  --method src-mst,src-spt,orc \
  --src_p 1.0 \
  --steps 20 \
  --trials 10 \
  --alpha 0.5 \
  --out ./results/real_src_orc.csv > logs/real_src_orc_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Other algorithms with `iGraph`

```sh
nohup python -u experiments/exp_igraph_real.py \
  --names karate,football,polbooks,polblogs,email-eu-core \
  --methods infomap,spinglass,fastgreedy,edgebetweenness,label_propagation \
  --trials 10 \
  --out ./results/real_igraph.csv > logs/real_igraph_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Summary script

```sh
python ./experiments/summary_real.py
```


## Curvature differences via root nodes on `Figure 3`

```sh
nohup python -u experiments/exp_sbm_root_diff.py \
  --method src-spt \
  --src_p 1.0 \
  --sbm-nodes 500 \
  --steps 20 \
  --ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --trials 10 \
  --src_root 0 \
  --out ./results/sbm_src_root_diff.csv > logs/sbm_src_root_diff$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

```sh
nohup python -u experiments/exp_lfr_root_diff.py \
  --lfr_nodes 500 \
  --method src-spt \
  --src_p 1.0 \
  --steps 20 \
  --mus 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --trials 10 \
  --out ./results/lfr_src_root_diff.csv > logs/lfr_src_root_diff$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Summary script

```sh
python ./experiments/summary_curv_diff.py
```


## Ablation of `p` on `Figure 4`

```sh
nohup python -u experiments/exp_p_ablation.py \
  --lfr_nodes 500 \
  --method src-spt \
  --src_p 1.0,1.5,2.0 \
  --steps 20 \
  --mus 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --trials 10 \
  --out ./results/lfr_src_p_ablation.csv > logs/lfr_src_p_ablation$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

```sh
python ./experiments/summary_lfr_p_ablation.py
```

## Ablation of `$alpha` on `Figure 5`

```sh
nohup python -u experiments/exp_lfr_alpha_ablation.py \
  --lfr_nodes 500 \
  --method src-spt,src-mst \
  --src_p 1.0 \
  --steps 20 \
  --mus 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --alpha 0.0,0.25,0.5,0.75,0.9 \
  --trials 10 \
  --out ./results/lfr_src_alpha.csv > logs/lfr_src_alpha_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

```sh
python ./experiments/summary_lfr_alpha_ablation.py
```

## Ablation of `$alpha` for `karate` and `football` on `Figure 6`

```sh
nohup python -u ./experiments/exp_real_alpha_ablation.py \
  --name karate,football \
  --method src-mst,src-spt \
  --src_p 1.0 \
  --steps 20 \
  --trials 10 \
  --alphas 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95 \
  --out ./results/real_alpha_ablation.csv > logs/real_alpha_ablation_spt_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

```sh
python ./experiments/summary_real_alpha_ablation.py
```

## Ablation of `beta` on `Figure 7`

```sh
nohup python -u experiments/exp_lfr_beta_ablation.py \
  --lfr_nodes 500 \
  --method src-spt \
  --src_p 1.0 \
  --steps 20 \
  --mus 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --betas 0.01,0.1,1,5,10\
  --trials 10 \
  --out ./results/lfr_src_beta.csv > logs/lfr_src_beta$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

```sh
python ./experiments/summary_lfr_beta_ablation.py
```

## Visualization of kappas on `Figure 8`

```sh
python ./experiments/summary_curvature_hist.py --mus 0.1 --steps 1 --method orc,src-spt,src-mst
```

```sh
python ./experiments/summary_curvature_hist.py --mus 0.4 --steps 1 --method orc,src-spt,src-mst
```

## Visualization on `Figure 9`

### `\mu=0.1`

```sh
python ./experiments/summary_lfr_random_edges_length.py --mu 0.1 --out ./results/lfr_random_src_0.1.pdf
```

### `\mu=0.4`

```sh
python ./experiments/summary_lfr_random_edges_length.py --mu 0.4 --out ./results/lfr_random_src_0.4.pdf
```

### `\mu=0.8`

```sh
python ./experiments/summary_lfr_random_edges_length.py --mu 0.8 --out ./results/lfr_random_src_0.8.pdf
```

## Visualization of root-depencence
### SBM

Compare SRC root-node dependence on SBM and generate `./results/sbm_src_orc_root_comparison.csv`.

```sh
nohup python -u experiments/exp_sbm_root_comparison.py \
  --method src-spt,src-mst,orc \
  --src_p 1.0 \
  --sbm-nodes 500 \
  --steps 50 \
  --ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --trials 10 \
  --src_roots 20 \
  --out ./results/sbm_src_orc_root_comparison.csv > logs/sbm_src_root_comparison$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

```sh
python experiments/summary_sbm_src_root_markdown.py
```


### LFR

Compare SRC root-node dependence on LFR and generate `./results/lfr_orc_src_root.csv`.

```sh
nohup python -u experiments/exp_lfr_src_root.py \
  --lfr_nodes 500 \
  --method src-spt,src-mst,orc \
  --src_p 1.0 \
  --steps 50 \
  --mus 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --src_roots 20 \
  --out ./results/lfr_orc_src_root.csv > logs/lfr_orc_src_root_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

```sh
python experiments/summary_lfr_src_root.py
```


## Generate Graphs

Generate cycle-rank-controlled evaluation graphs (manifest + GraphML).

```sh
python experiments/generate_cycle_controlled_graphs.py \
  --dataset grid \
  --grid-size 20 \
  --noise 0.02 \
  --k 8 \
  --etas 0 0.01 0.02 0.05 0.10 0.20 0.50 1.0 \
  --add-mode short \
  --seed 0 \
  --output-dir outputs/cycle_controlled_grid
```

## Generate SRC(MST) Evaluation

Run SRC-vs-ORC gap evaluation with MST projection on the generated graphs,
and output `orc_src_gap_summary.csv` / `orc_src_edge_curvatures.csv`.

```sh
python experiments/evaluate_orc_src_gap_with_edge_selection_counts.py \
  --graph-dir outputs/cycle_controlled_grid \
  --edge-set tree \
  --src-method src-mst \
  --src-root 0 \
  --src-delta 0.5 \
  --src-p 1.0 \
  --orc-alpha 0.5 \
  --orc-proc 1 \
  --output-dir outputs/cycle_controlled_grid/orc_src_gap_mst
```

## Generate SRC(SPT) Evaluation

Run SRC-vs-ORC gap evaluation with SPT projection on the generated graphs,
and output `orc_src_gap_summary.csv` / `orc_src_edge_curvatures.csv`.

```sh
python experiments/evaluate_orc_src_gap_with_edge_selection_counts.py \
  --graph-dir outputs/cycle_controlled_grid \
  --edge-set tree \
  --src-method src-spt \
  --src-root 0 \
  --src-delta 0.5 \
  --src-p 1.0 \
  --orc-alpha 0.5 \
  --orc-proc 1 \
  --output-dir outputs/cycle_controlled_grid/orc_src_gap_spt
```
### Using $\eta$ as the X-axis

Combine MST/SPT summary CSV files and generate comparison plots of the ORC-SRC gap against $\eta$.

```sh
python experiments/plot_orc_src_gap_compare_methods_with_beta1.py \
  --mst-summary outputs/cycle_controlled_grid/orc_src_gap_mst/orc_src_gap_summary.csv \
  --spt-summary outputs/cycle_controlled_grid/orc_src_gap_spt/orc_src_gap_summary.csv \
  --x eta_actual \
  --combined-only \
  --output-dir outputs/orc_src_gap_compare_eta
```

```sh
python experiments/plot_orc_src_histograms_compare_methods.py \
  --mst-edge-csv outputs/cycle_controlled_grid/orc_src_gap_mst/orc_src_edge_curvatures.csv \
  --spt-edge-csv outputs/cycle_controlled_grid/orc_src_gap_spt/orc_src_edge_curvatures.csv \
  --etas 0 0.2 0.5 1.0 \
  --edge-set tree \
  --bins 50 \
  --xlim -8 1. \
  --output-dir outputs/cycle_controlled_grid/orc_src_gap_compare_hist
```

