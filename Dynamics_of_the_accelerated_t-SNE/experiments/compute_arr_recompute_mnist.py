import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import jv, iv

import utils


METHODS = ("GD", "MM", "NAG")


def parse_args():
    """
    Parse command-line options for ARR experiments.

    - inputs:
        - CLI arguments from the current process.
    - outputs:
        - argparse.Namespace with experiment configuration.
    - process:
        - Define parser options and return parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Recompute MNIST data/P and calculate ARR stopping times for continuous EE relaxations."
    )

    parser.add_argument("--labels", nargs="+", default=["2", "4", "6", "8"])
    parser.add_argument("--sample-number", type=int, default=400)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=4.0)

    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--initial-beta-coefficient", type=float, default=1e-4)
    parser.add_argument("--initialization-std", type=float, default=1e-4)

    parser.add_argument("--h-values", nargs="+", type=float, default=[100.0, 200.0, 400.0])
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--k-max", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=0.01)

    parser.add_argument(
        "--constant-mode",
        choices=["formula", "current_utils"],
        default="formula",
        help=(
            "formula: keep the first Laplacian eigenmode as a translation-invariant constant mode. "
            "current_utils: mimic the current utils.py treatment of w[0]."
        ),
    )

    parser.add_argument("--n-neighbors-tw", type=int, default=30)
    parser.add_argument("--output-dir", default="outputs/arr_recompute_mnist")

    return parser.parse_args()


def load_data_and_probabilities(args):
    """
    Load MNIST subset, compute t-SNE probabilities, and initialize Y.

    - inputs:
        - Parsed CLI args with data and probability settings.
    - outputs:
        - X sampled array, labels array, probability matrix P, and initial Y.
    - process:
        - Build original_tSNE, load data, compute P, and sample Gaussian initial points.
    """
    base = utils.original_tSNE(
        labels=args.labels,
        sample_data="mnist",
        perplexity=args.perplexity,
        sample_number=args.sample_number,
        total_iteration=1000,
        ee_iteration=50,
        alpha=args.alpha,
        learning_rate=100.0,
        initial_momentum=0.5,
        final_momentum=0.8,
        momentum_switch_iteration=250,
        pca_components=args.pca_components,
        initial_beta_coefficient=args.initial_beta_coefficient,
        tol=args.tol,
        random_state=args.seed,
        initialization_std=args.initialization_std,
    )

    base.load_original_mnist()
    base.tsne_probabilities(
        X=base.X_sampled,
        perplexity=base.perplexity,
        tol=base.tol,
        initial_beta_coefficient=base.initial_beta_coefficient,
    )

    rng = np.random.RandomState(args.seed)
    initial_Y = rng.normal(
        loc=0.0,
        scale=args.initialization_std,
        size=(base.N, 2),
    )

    return base.X_sampled, np.array(base.y_sampled), base.P.copy(), initial_Y


def build_spectral_system(P, initial_Y, alpha, constant_mode):
    """
    Build the Laplacian spectral system used for ARR evolution.

    - inputs:
        - Probability matrix P, initial embedding Y(0), alpha, and constant-mode policy.
    - outputs:
        - Eigenvalues, eigenvectors, effective spectrum w, and initial coefficients coeff0.
    - process:
        - Compute graph Laplacian, eigendecomposition, transform Y into spectral coefficients,
          and adjust the constant mode according to the selected policy.
    """
    n = P.shape[0]

    D = np.diag(P.sum(axis=1))
    L = D - P

    eigvals, eigvecs = np.linalg.eigh(L)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    w = alpha * eigvals - 1.0 / (n - 1)

    if constant_mode == "formula":
        # Translation mode should remain constant.
        w[0] = 0.0

    elif constant_mode == "current_utils":
        # Mimic current utils.py behavior as closely as possible.
        # In utils.py, after w = alpha*w - 1/(N-1),
        # w[0] is overwritten as alpha*w[0] + 1/(N-1).
        w[0] = alpha * w[0] + 1.0 / (n - 1)

    coeff0 = eigvecs.T @ initial_Y

    return eigvals, eigvecs, w, coeff0


def nag_scales(t, w):
    """
    Compute continuous-time NAG scaling factors.

    - inputs:
        - Time value t and spectral values w.
    - outputs:
        - Array of per-mode scale factors.
    - process:
        - Use Bessel J1 for positive modes, modified Bessel I1 for negative modes,
          and return ones for zero-time/zero-mode cases.
    """
    w = np.asarray(w)
    scales = np.ones_like(w, dtype=float)

    if t == 0:
        return scales

    positive = w > 0
    negative = w < 0

    if np.any(positive):
        z = t * np.sqrt(w[positive])
        scales[positive] = 2.0 / z * jv(1, z)

    if np.any(negative):
        z = t * np.sqrt(-w[negative])
        scales[negative] = 2.0 / z * iv(1, z)

    return scales


def coefficients_at_time(coeff0, w, method, h, momentum, k):
    """
    Compute spectral coefficients at iteration/time for one method.

    - inputs:
        - Initial coefficients, spectral values, method, step h, momentum, and iteration k.
    - outputs:
        - Continuous-time value t and propagated coefficients coeff_t.
    - process:
        - Select method-specific scaling rule (GD/MM/NAG) and apply it to coeff0.
    """
    if method == "GD":
        t = k * h
        scales = np.exp(-t * w)

    elif method == "MM":
        t = k * h
        scales = np.exp(-t * w / (1.0 - momentum))

    elif method == "NAG":
        t = k * np.sqrt(h)
        scales = nag_scales(t, w)

    else:
        raise ValueError(f"Unknown method: {method}")

    coeff_t = scales[:, None] * coeff0
    return t, coeff_t


def calc_arr(coeff_t, R):
    """
    Compute ARR from spectral coefficients.

    - inputs:
        - Coefficients coeff_t and head-mode count R.
    - outputs:
        - ARR scalar value (or NaN if denominator is non-positive).
    - process:
        - Aggregate absolute coefficients by mode, split head/tail means,
          and return tail ratio over total mean mass.
    """
    n = coeff_t.shape[0]
    a = np.sum(np.abs(coeff_t), axis=1)

    head_mean = np.sum(a[:R]) / R
    tail_mean = np.sum(a[R:]) / max(n - R, 1)

    denom = head_mean + tail_mean
    if denom <= 0:
        return np.nan

    return tail_mean / denom


def compute_arr_table(coeff0, w, R, h, momentum, k_max):
    """
    Build ARR trajectories over iterations for all methods.

    - inputs:
        - Initial coefficients, spectrum w, head count R, h, momentum, and k_max.
    - outputs:
        - DataFrame with method-wise ARR values across k and t.
    - process:
        - Loop over methods/iterations, propagate coefficients, compute ARR, and collect rows.
    """
    rows = []

    for method in METHODS:
        for k in range(k_max + 1):
            t, coeff_t = coefficients_at_time(
                coeff0=coeff0,
                w=w,
                method=method,
                h=h,
                momentum=momentum,
                k=k,
            )
            arr = calc_arr(coeff_t, R=R)

            rows.append(
                {
                    "method": method,
                    "h": h,
                    "momentum": momentum,
                    "k": k,
                    "t": t,
                    "ARR": arr,
                }
            )

    return pd.DataFrame(rows)


def summarize_threshold(df, epsilon):
    """
    Summarize first ARR threshold hit per method.

    - inputs:
        - ARR DataFrame and threshold epsilon.
    - outputs:
        - DataFrame with first reached (k, t, ARR) and reach flags per method.
    - process:
        - Filter rows by ARR <= epsilon, select earliest event, and build summary rows.
    """
    rows = []

    for method in METHODS:
        tmp = df[(df["method"] == method) & (df["ARR"] <= epsilon)].copy()

        if tmp.empty:
            rows.append(
                {
                    "method": method,
                    "h": df["h"].iloc[0],
                    "k": None,
                    "t": None,
                    "ARR": None,
                    "epsilon": epsilon,
                    "reached": False,
                }
            )
            continue

        first = tmp.sort_values(["t", "k"]).iloc[0]
        rows.append(
            {
                "method": method,
                "h": first["h"],
                "k": int(first["k"]),
                "t": float(first["t"]),
                "ARR": float(first["ARR"]),
                "epsilon": epsilon,
                "reached": True,
            }
        )

    return pd.DataFrame(rows)


def plot_arr(df, summary, epsilon, output_path):
    """
    Plot ARR curves and threshold-hit annotations.

    - inputs:
        - ARR DataFrame, threshold summary DataFrame, epsilon, and output path.
    - outputs:
        - Saved ARR figure file.
    - process:
        - Draw method curves, threshold line, reached markers/text, and save the plot.
    """
    plt.figure(figsize=(9, 5))

    for method in METHODS:
        tmp = df[df["method"] == method]
        plt.plot(tmp["t"], tmp["ARR"], label=method)

    plt.axhline(epsilon, linestyle="--", linewidth=1.0, label=f"ARR={epsilon}")

    for _, row in summary.iterrows():
        if bool(row["reached"]):
            plt.scatter(row["t"], row["ARR"], s=40)
            plt.text(
                row["t"],
                row["ARR"],
                f" {row['method']}: k={int(row['k'])}, t={row['t']:.2f}",
                fontsize=8,
                va="bottom",
            )

    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel("ARR")
    plt.title(f"ARR transition (h={df['h'].iloc[0]:g})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    """
    Run the complete ARR recomputation pipeline.

    - inputs:
        - CLI arguments from the current process.
    - outputs:
        - ARR CSVs, threshold CSVs, plots, and console summaries.
    - process:
        - Load data/P, build spectral system, compute ARR per h, summarize, and write outputs.
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    X, labels, P, initial_Y = load_data_and_probabilities(args)

    R = len(np.unique(labels))
    n = P.shape[0]

    print("Data loaded")
    print("  X:", X.shape)
    print("  P:", P.shape)
    print("  initial_Y:", initial_Y.shape)
    print("  labels:", np.unique(labels))
    print("  n:", n)
    print("  R:", R)
    print("  alpha:", args.alpha)
    print("  momentum:", args.momentum)
    print("  constant_mode:", args.constant_mode)

    eigvals, eigvecs, w, coeff0 = build_spectral_system(
        P=P,
        initial_Y=initial_Y,
        alpha=args.alpha,
        constant_mode=args.constant_mode,
    )

    print("\nSpectral summary")
    print("  eigvals[:10]:", eigvals[:10])
    print("  w[:10]:", w[:10])
    print("  min(w):", np.min(w))
    print("  max(w):", np.max(w))

    all_summaries = []

    for h in args.h_values:
        print(f"\nComputing ARR for h={h:g}")

        df = compute_arr_table(
            coeff0=coeff0,
            w=w,
            R=R,
            h=h,
            momentum=args.momentum,
            k_max=args.k_max,
        )

        summary = summarize_threshold(df, epsilon=args.epsilon)
        all_summaries.append(summary)

        arr_csv = os.path.join(
            args.output_dir,
            f"arr_curve_h{h:g}_{args.constant_mode}.csv",
        )
        summary_csv = os.path.join(
            args.output_dir,
            f"arr_threshold_h{h:g}_{args.constant_mode}.csv",
        )
        fig_path = os.path.join(
            args.output_dir,
            f"arr_curve_h{h:g}_{args.constant_mode}.png",
        )

        df.to_csv(arr_csv, index=False)
        summary.to_csv(summary_csv, index=False)
        plot_arr(df, summary, args.epsilon, fig_path)

        print(summary.to_string(index=False))
        print("Saved:", arr_csv)
        print("Saved:", summary_csv)
        print("Saved:", fig_path)

    all_summary = pd.concat(all_summaries, ignore_index=True)
    all_summary_csv = os.path.join(
        args.output_dir,
        f"arr_threshold_all_h_{args.constant_mode}.csv",
    )
    all_summary.to_csv(all_summary_csv, index=False)

    print("\nAll threshold summary")
    print(all_summary.to_string(index=False))
    print("Saved:", all_summary_csv)


if __name__ == "__main__":
    main()