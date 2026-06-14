
# MANL features for Sobolev-Ricci Curvature

This subdirectory contains code and notebooks of MANL features used in the experiments.

The study explores how **Ollivier–Ricci Curvature (ORC)** and its variants can be incorporated into graph-based learning tasks.  
We investigate both **theoretical properties** (connections between curvature and robustness/generalization) and **practical algorithms**, including edge-pruning strategies.

The implementation builds upon [`orcml`](https://github.com/TristanSaidi/orcml), which is included in this repo under [./orcml](./orcml/).
 
---

## Setup

Please set-up `venv` setup as follows:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r ./requirements.txt
```
---

## Notebooks

---

### Experiments with Edge-Pruning
These notebooks focus on **edge-pruning strategies** motivated by curvatures.

- [b01_edge_pruning.ipynb](./b01_edge_pruning.ipynb):  
  Generates **Figure 2** and **Figure 12** for data generation, synthetic experiments, and pruning visualization.
  This notebook generates the both 2D/3D data cloud, which are loaded in executing the [b02_vizualize_2d_3d.ipynb](./b02_vizualize_2d_3d.ipynb) and [b03_grid_search_manl.ipynb](./b03_grid_search_manl.ipynb) notebooks.

- [b02_vizualize_2d_3d.ipynb](./b02_vizualize_2d_3d.ipynb):  
  Visualize the point cloud such as **Figures 10** and **11**.

- [b03_grid_search_manl.ipynb](./b03_grid_search_manl.ipynb)
  Supplementary notebook for grid-search experimentation in deciding `\delta_M` and `\lambda_M`.
