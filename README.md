# Geometric Perspectives on Graph-based Learning Algorithms

This repository contains supplementary materials for the PhD thesis titled
"Geometric perspectives on graph-based learning algorithms".

## Summary

1. Dynamics of the accelerated t-SNE
    - The first part is based on the [Dynamics of the accelerated t-SNE](https://openreview.net/forum?id=dfUebM9asV) submitted to and reviewed by [TMLR (official)](https://jmlr.org/tmlr/).
    - In addition to the reviewed content, this repository includes several extra experiments. The corresponding implementation and runnable scripts are placed under the [Dynamics_of_the_accelerated_t-SNE](./Dynamics_of_the_accelerated_t-SNE), with experiment scripts mainly organized in [Dynamics_of_the_accelerated_t-SNE/experiments](./Dynamics_of_the_accelerated_t-SNE/experiments).

2. Sobolev Ricci Curvature
    - The second part focuses on Sobolev Ricci Curvature, which extracts tree structures from graphs and is closely related to Ollivier Ricci Curvature. This work is available as a preprint at　[Sobolev--Ricci Curvature](https://arxiv.org/abs/2603.12652)
    - Implementation and experiment scripts are organized under the [SobolevRicciCurvature](./SobolevRicciCurvature). For application-oriented tasks, please check the two main artifacts:
        - [SobolevRicciCurvature/community_detection](./SobolevRicciCurvature/community_detection)
        - [SobolevRicciCurvature/manl](./SobolevRicciCurvature/manl)


## Repository structure

| Directory | Description |
|---|---|
| `Dynamics_of_the_accelerated_t-SNE/` | Code and notebooks for the accelerated t-SNE experiments, including GD/MM/NAG comparisons, ARR computation, and additional experiments for the dissertation. |
| `Dynamics_of_the_accelerated_t-SNE/experiments/` | Scripts for ARR-stopped continuous relaxations, comparison with discrete t-SNE, KL-divergence trajectories, and momentum sweep experiments. |
| `SobolevRicciCurvature/community_detection/` | Community-detection experiments using SRC/ORC, including SBM, LFR, real graph datasets, ablation studies, and cycle-controlled ORC--SRC discrepancy experiments. |
| `SobolevRicciCurvature/manl/` | MANL / edge-pruning experiments using ORC and SRC variants. |

## Reproducing dissertation experiments

The code is organized by dissertation topic.

- For Chapter 3, see `Dynamics_of_the_accelerated_t-SNE/experiments/README_tsne.md`.
  This includes ARR computation, ARR-stopped continuous relaxations, comparison with the original discrete t-SNE update, KL-divergence trajectories, and momentum-sweep experiments.
- For Chapter 4 community-detection experiments, see `SobolevRicciCurvature/community_detection/README.md`.
- For Chapter 4 MANL / edge-pruning experiments, see `SobolevRicciCurvature/manl/README.md`.