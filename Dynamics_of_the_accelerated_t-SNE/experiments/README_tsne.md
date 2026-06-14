# Additional experiments

This directory contains additional experiments beyond the paper, prepared for the thesis to provide deeper empirical evidence on accelerated t-SNE dynamics.
The scripts focus on comparing discrete and continuous formulations, checking robustness across optimization choices, and visualizing how differences appear at the EE stage and after full embedding optimization.


## Setup

Create an isolated Python environment and install dependencies before running the experiments below.

Please set-up `venv` setup as follows:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
pip3 install -r ./requirements.txt
```

## Additional experiments for the dissertation

This directory contains scripts used for additional experiments in Chapter 3 and Appendix A.

| Script | Purpose |
|---|---|
| `compute_arr_recompute_mnist.py` | Computes ARR trajectories for MNIST digits used in the stopping-time experiment. |
| `compute_embedding_stage_gd_mm_nag_h400_only.py` | Compares ARR-stopped continuous relaxations with discrete t-SNE after the common embedding stage. |
| `plot_embedding_stage_gd_mm_nag_h400_only.py` | Generates the embedding comparison figures. |
| `compute_kl_divergence_mm.py` | Computes KL-divergence trajectories for the comparison experiments. |
| `plot_kl_divergence_h400_only.py` | Generates the KL-divergence trajectory figures. |
| `compute_discrete_mm_momentum_sweep.py` | Sweeps the EE-stage momentum parameter in the original discrete t-SNE update. |
| `plot_discrete_mm_momentum_sweep.py` | Generates the corresponding embedding and KL-divergence plots. |


## Calculate ARR(=Average Residual Ratio)

This experiment measures ARR to quantify residual behavior over iterations under different continuous-time step settings (`h-values`).
Use it when you want a compact numerical comparison of convergence behavior before plotting embeddings.

```sh
python3 compute_arr_recompute_mnist.py \
  --labels 2 4 6 8 \
  --sample-number 400 \
  --perplexity 30 \
  --alpha 4 \
  --pca-components 30 \
  --seed 0 \
  --h-values 100 200 400 \
  --momentum 0.5 \
  --k-max 300 \
  --epsilon 0.005 \
  --constant-mode current_utils \
  --output-dir outputs/arr_recompute_mnist_current_utils
```

## Calculate both on EE and embedding stages

This experiment computes and stores outputs for both milestones: the end of the early exaggeration (EE) stage and the final embedding stage.
It is useful for analyzing whether differences introduced during EE persist or diminish after the remaining optimization.

```sh
python3 compute_embedding_stage_gd_mm_nag_h400_only.py \
  --labels 2 4 6 8 \
  --sample-number 400 \
  --perplexity 30 \
  --ee-iterations 50 \
  --total-iterations 1000 \
  --alpha 4 \
  --discrete-learning-rate 100 \
  --embedding-learning-rate 100 \
  --continuous-h-values 400 \
  --continuous-h400-ee-iterations 12 6 13 \
  --pca-components 30 \
  --seed 0 \
  --embedding-initial-momentum 0.5 \
  --embedding-final-momentum 0.8 \
  --embedding-momentum-switch-iteration 250 \
  --output-dir outputs/comparison_embedding_stage_gd_mm_nag_h400_only
```

## Visualize the results on EE and embedding stages

This script reads the saved experiment outputs and generates side-by-side figures for GD/MM/NAG across discrete and continuous settings.
Use these plots to visually inspect structural similarities and differences at each stage.

```sh
python3 plot_embedding_stage_gd_mm_nag_h400_only.py \
  --npz outputs/comparison_embedding_stage_gd_mm_nag_h400_only/comparison_embedding_stage_gd_mm_nag_h400_only.npz \
  --output-dir outputs/comparison_embedding_stage_gd_mm_nag_h400_eps0005
```

This command reads the `.npz` file and generates two comparison figures: one at the EE endpoint and one after the common embedding stage.
It is useful for checking whether visible differences at EE remain after full optimization.

```sh
python3 plot_kl_divergence_h400_only.py \
  --costs-csv outputs/comparison_embedding_stage_gd_mm_nag_h400_only/costs_h400_only.csv \
  --output-dir outputs/comparison_embedding_stage_gd_mm_nag_h400_eps0005
```

This command visualizes KL-divergence trajectories by `k` for the same experiment outputs.
The left panel shows the EE stage only, while the right panel summarizes EE+embedding behavior in one view.


## Additional experiments with various momentum coefficients

This sweep evaluates how different momentum values in the discrete MM setting affect trajectories and final embeddings.
It provides a sensitivity analysis of momentum-dependent behavior beyond the default configuration.

```sh
python3 compute_discrete_mm_momentum_sweep.py \
  --labels 2 4 6 8 \
  --sample-number 400 \
  --perplexity 30 \
  --ee-iterations 50 \
  --total-iterations 1000 \
  --alpha 4.0 \
  --ee-learning-rate 100 \
  --ee-momentums 0.0 0.1 0.3 0.5 0.7 0.9 \
  --embedding-learning-rate 100 \
  --embedding-initial-momentum 0.5 \
  --embedding-final-momentum 0.8 \
  --embedding-momentum-switch-iteration 250 \
  --seed 0 \
  --output-dir outputs/discrete_mm_momentum_sweep
```

### Visualization of the results

These plotting commands summarize the momentum sweep results with different axis-scaling options.
They help compare shape changes consistently across panels or under shared global limits.

```sh
python3 plot_discrete_mm_momentum_sweep.py \
  --npz outputs/discrete_mm_momentum_sweep/discrete_mm_momentum_sweep.npz \
  --output-dir outputs/discrete_mm_momentum_sweep \
  --ee-scale panel \
  --embedding-scale global \
  --equal-each-panel
```

## Visualize KL-divergence 

This utility computes and reports KL-divergence values for the EE stage across multiple momentum coefficients.
In this setup, momentum is varied only during the EE stage, while the subsequent embedding stage is run with a unified algorithmic setting for fair comparison.
The primary goal is to verify how KL-divergence changes with momentum settings and to quantify differences observed in the momentum sweep.

```sh
python3 compute_kl_divergence_mm.py
```


## Additional momentum

This setting increases EE-stage iterations in the continuous h=400 comparison and is intended to stress-test how trajectory gaps change under a longer EE regime.
Use it when you want to compare the default setup against a more aggressive EE schedule while keeping the embedding stage configuration fixed.

```sh
python3 compute_embedding_stage_gd_mm_nag_h400_only.py \
  --labels 2 4 6 8 \
  --sample-number 400 \
  --perplexity 30 \
  --ee-iterations 50 \
  --total-iterations 1000 \
  --alpha 4 \
  --discrete-learning-rate 100 \
  --embedding-learning-rate 100 \
  --continuous-h-values 400 \
  --continuous-h400-ee-iterations 30 30 30 \
  --pca-components 30 \
  --seed 0 \
  --embedding-initial-momentum 0.5 \
  --embedding-final-momentum 0.8 \
  --embedding-momentum-switch-iteration 250 \
  --output-dir outputs/comparison_embedding_stage_gd_mm_nag_h400_only_high_h
```

## Visualize additional momentum results

After computation, generate both geometric and cost-based diagnostics for the high-h outputs.
The first plot script creates embedding snapshots; the second creates KL-divergence trajectories from `costs_h400_only.csv`.

```sh
python3 plot_embedding_stage_gd_mm_nag_h400_only.py \
  --npz outputs/comparison_embedding_stage_gd_mm_nag_h400_only_high_h/comparison_embedding_stage_gd_mm_nag_h400_only.npz \
  --output-dir outputs/comparison_embedding_stage_gd_mm_nag_h400_high_h
```

```sh
python3 plot_kl_divergence_h400_only.py \
  --costs-csv outputs/comparison_embedding_stage_gd_mm_nag_h400_only_high_h/costs_h400_only.csv \
  --output-dir outputs/comparison_embedding_stage_gd_mm_nag_h400_high_h
```

