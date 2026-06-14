## The purpose of this repository

- This repository is prepared as the supplementary material with the paper titled [Dynamics of the accelerated t-SNE](https://openreview.net/forum?id=dfUebM9asV).
- The paper introduces Momentum methods (MM) and Nesterov Accelerated Gradient (NAG) to the t-SNE algorithm with the solution functions of ordinary differential equations such as exponential functions or (modified) Bessel functions of the first kind. MM and NAG can be interpreted as an extention of Gradient Descent (GD) method.
- For additional experiments, please see [./experiments/README_tsne.md](./experiments/README_tsne.md).

## How to use the contents

- This repository requires you to use `python`, `jupyter notebook` and some libraries defined in [requirements.txt](./requirements.txt). If you don't have Jupyter Notebook, you can install them by following the instructions in the [official website](https://jupyter.org/install).

## The contents
- Note: You can find some hyperlinks to the referenced PDF and GIF files in the paper.
- The repository contains the following files:
  - [1.0 SolutionPath with KDDCup1999](./1.0_SolutionPath_with_KDDCupData.ipynb): An experimentation with KDDCup99 data with GD/MM/NAG. 
    - Includes 
      - Heatmap of adjacency matrix `P`.
      - Distribution of eigenvalues and eigenvectors.
      - Average Residual Ratio(ARR) and its visualization with [arr_plot_kddcup1999.pdf](./arr_plot_kddcup1999.pdf) and [scatter_plot_50_25_33.pdf](./scatter_plot_50_25_33.pdf).
      - Comparison of the low-dimensional representation among GD/MM/NAG.
      - Experiments of various random initialization with [ari_plot_kddcup.pdf](./ari_plot_kddcup.pdf)
    - We can find the dataset in [here](./data/kddcup.data_10_percent.gz), or [original KDDCup99 data](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) is also available.
  - [1.1 SolutionPath with GMM Data](./1.1_SolutionPath_with_GMMData.ipynb): An experimentation with Gaussian mixture synthetic data with GD/MM/NAG, and an evaluation of cost functions (KL-divergence).
    - Includes 
        - Heatmap of adjacency matrix `P`
        - distribution of eigenvectorss and eigenvalues
        - Comparison of the low-dimensional representation among GD/MM/NAG
        - Evaluation of cost functions(KL-divergence) for GD_ODE/GD_iterative/MM_ODE/MM_iterative/NAG_ODE/NAG_iterative
  - [1.2 SolutionPath with MNIST](./1.2_SolutionPath_with_MNISTData.ipynb): An experimentation with MNIST data.
    - Includes
        - Heatmap of adjacency matrix `P`
        - Distribution of eigenvectors
        - Average Residual Ratio(ARR) and its visualization with [arr_plot_mnist.pdf](./arr_plot_mnist.pdf) and [scatter_plot_27_14_28_with_embedding_200.pdf](./scatter_plot_27_14_28_with_embedding_200.pdf).
        - ARR with various momentum coefficients with [arr_plot_mnist_mm.pdf](./arr_plot_mnist_mm.pdf).
        - Experiments of various random initialization with [ari_plot_mnist.pdf](./ari_plot_mnist.pdf)
        - Visualize experiments results with various initialization with [ari_plot_mnist_initialize.pdf](./ari_plot_mnist_initialize.pdf)
        - A [video](./mnist_clustering_200.gif) of the cluster formation process for GD, MM, and NAG when applying the iterative method.
  - [1.3 SolutionPath with Olivetti Faces](./1.3_SolutionPath_with_Olivetti_Face.ipynb): An experimentation with Olivetti Faces dataset.
    -  Includes
        - Heatmap of adjacency matrix `P`
        - Distribution of eigenvectors.
        - Average Residual Ratio(ARR) and its visualization with [arr_plot_olivetti.pdf](./arr_plot_olivetti.pdf) and [scatter_plot_103_52_61.pdf](./scatter_plot_103_52_61.pdf).
        - Experiments of various random initialization with [ari_plot_olivetti.pdf](./ari_plot_olivetti.pdf)
  - [2.0 Comparison of functions: Exponential and Bessel](./2.0_Comparison_of_functions.ipynb): An experimentation with the comparison of exponential and (modified) Bessel functions.
    - Includes numerical transition simulations about the cases on both negative and positive eigenvalues.
  - [3.0_basic_tSNE_implementation](./3.0_basic_tSNE_implementation.ipynb): An experiment with basic operation with t-SNE algorithm. You can use Gradient Descent, Moment Method, and Nesterov Accelerated Gradient methods.
  - [utils.py](./utils.py): Utility functions, which include actual implementation of the accelerated t-SNE.
  - `requirements.txt`: The text file that contains the required packages to run the Jupyter Notebook file. Please use the following command:
    ```sh
    pip install -r ./requirements.txt
    ```

## Related paper
  Please find the [Openreview Link](https://openreview.net/forum?id=dfUebM9asV) for detailed contents, including the background theory.

## Used datasets
- [KDDCup1999](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [MNIST](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
- [Olivetti Faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html)