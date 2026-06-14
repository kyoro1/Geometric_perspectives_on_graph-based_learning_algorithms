#!/usr/bin/env python3
"""
Compare discrete momentum coefficients in the early exaggeration (EE) stage.

For each EE momentum coefficient in
  0.0 (= GD), 0.1, 0.3, 0.5, 0.7, 0.9

this script:
  1. runs the discrete t-SNE EE stage with MM updates and the specified
     constant momentum coefficient;
  2. starts from the resulting EE endpoint;
  3. runs the original t-SNE embedding stage with the usual momentum schedule.

Outputs:
  - discrete_mm_momentum_sweep.npz
  - summary_discrete_mm_momentum_sweep.csv
  - costs_discrete_mm_momentum_sweep.csv

This is meant to isolate the effect of the discrete EE-stage momentum coefficient.
The embedding stage is kept fixed across all variants.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import utils


def parse_args():
    """
    Parse command-line options for the discrete MM momentum sweep.

    - inputs:
        - CLI arguments from the current process.
    - outputs:
        - argparse.Namespace with sweep and output settings.
    - process:
        - Define parser flags for EE/embedding configs and return parsed args.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run discrete EE-stage MM with several constant momentum coefficients, "
            "then run the original t-SNE embedding stage from each EE endpoint."
        )
    )

    parser.add_argument("--labels", nargs="+", default=["2", "4", "6", "8"])
    parser.add_argument("--sample-number", type=int, default=400)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--ee-iterations", type=int, default=50)
    parser.add_argument("--total-iterations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=4.0)

    # EE-stage settings.
    parser.add_argument("--ee-learning-rate", type=float, default=100.0)
    parser.add_argument(
        "--ee-momentums",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        help=(
            "Constant momentum coefficients used only in the EE stage. "
            "0.0 corresponds to GD."
        ),
    )

    # Embedding-stage settings: original t-SNE schedule.
    parser.add_argument("--embedding-learning-rate", type=float, default=100.0)
    parser.add_argument("--embedding-initial-momentum", type=float, default=0.5)
    parser.add_argument("--embedding-final-momentum", type=float, default=0.8)
    parser.add_argument("--embedding-momentum-switch-iteration", type=int, default=250)

    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--initial-beta-coefficient", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="outputs/discrete_mm_momentum_sweep",
    )

    return parser.parse_args()


def momentum_label(momentum: float) -> str:
    """
    Convert a momentum value into a file-safe label.

    - inputs:
        - Momentum coefficient as float.
    - outputs:
        - Normalized string label (e.g., m0p5).
    - process:
        - Format to one decimal and replace special characters.
    """
    return f"m{momentum:.1f}".replace(".", "p").replace("-", "neg")


def build_data_and_probabilities(args):
    """
    Prepare base dataset and t-SNE probabilities.

    - inputs:
        - Parsed CLI args with data and optimization settings.
    - outputs:
        - Initialized original_tSNE object containing sampled data and P.
    - process:
        - Instantiate original_tSNE, load MNIST subset, and compute probability matrix.
    """
    base = utils.original_tSNE(
        labels=args.labels,
        sample_data="mnist",
        perplexity=args.perplexity,
        sample_number=args.sample_number,
        total_iteration=args.total_iterations,
        ee_iteration=args.ee_iterations,
        alpha=args.alpha,
        learning_rate=args.embedding_learning_rate,
        initial_momentum=args.embedding_initial_momentum,
        final_momentum=args.embedding_final_momentum,
        momentum_switch_iteration=args.embedding_momentum_switch_iteration,
        pca_components=args.pca_components,
        initial_beta_coefficient=args.initial_beta_coefficient,
        tol=args.tol,
        random_state=args.seed,
    )

    base.load_original_mnist()
    base.tsne_probabilities(
        X=base.X_sampled,
        perplexity=base.perplexity,
        tol=base.tol,
        initial_beta_coefficient=base.initial_beta_coefficient,
    )
    return base


def _run_embedding_stage_from_state(
    embedding_runner,
    args,
    Y_ee,
    log_prefix,
    ee_iteration,
):
    """
    Run the common embedding stage from a selected EE endpoint.

    - inputs:
        - Embedding runner, args, EE endpoint Y_ee, log prefix, and EE iteration index.
    - outputs:
        - Final embedding array and embedding-stage cost trajectory.
    - process:
        - Reset state buffers/velocity at EE boundary, run MM updates for embedding stage,
          collect costs, and return final state.
    """
    embedding_runner._log_prefix = log_prefix
    embedding_runner._log_stage = "embedding"
    embedding_runner._log_interval = args.log_interval

    embedding_runner.Y = {
        ee_iteration - 1: Y_ee.copy(),
        ee_iteration: Y_ee.copy(),
    }
    embedding_runner.Y_nes = {ee_iteration: Y_ee.copy()}
    embedding_runner.Q = dict()
    embedding_runner.S = dict()
    embedding_runner.cost = dict()
    embedding_runner.squared_distances = dict()
    embedding_runner.student_kernel = dict()

    alpha_original = embedding_runner.alpha
    embedding_runner.alpha = 1.0
    embedding_costs = []

    print(f"[{log_prefix}] stage start: embedding ({ee_iteration} to {args.total_iterations - 1})")
    for k in range(ee_iteration, args.total_iterations):
        embedding_runner.momentum = embedding_runner._stage_momentum(k + 1, "MM")
        embedding_runner.from_Y_to_Q(k=k)
        embedding_runner.from_Q_to_S(k=k)
        embedding_runner.from_S_to_Y(k=k, optimization_method="MM")
        embedding_costs.append(embedding_runner.cost[k])

        # Keep memory small.
        for buf in (
            embedding_runner.Q,
            embedding_runner.S,
            embedding_runner.squared_distances,
            embedding_runner.student_kernel,
        ):
            buf.pop(k, None)

        if k - 1 >= ee_iteration and (k - 1) in embedding_runner.Y:
            del embedding_runner.Y[k - 1]

    embedding_runner.alpha = alpha_original
    final_Y = embedding_runner.Y[args.total_iterations].copy()
    print(f"[{log_prefix}] stage end")
    return final_Y, np.array(embedding_costs)


def _summary_stats(Y):
    """
    Compute compact geometric statistics for one embedding.

    - inputs:
        - Embedding array Y.
    - outputs:
        - Dict with std_x, std_y, max_abs, and anisotropy.
    - process:
        - Center data, estimate covariance spectrum, and derive scalar summaries.
    """
    centered = Y - Y.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / Y.shape[0]
    eig = np.linalg.eigvalsh(cov)
    return {
        "std_x": float(Y[:, 0].std()),
        "std_y": float(Y[:, 1].std()),
        "max_abs": float(np.max(np.abs(Y))),
        "anisotropy": float(eig[-1] / (eig[0] + 1e-300)),
    }


def run_all_variants(args, base, initial_Y):
    """
    Execute EE momentum sweep and common embedding continuation.

    - inputs:
        - Parsed args, prepared base object, and shared initial Y.
    - outputs:
        - EE states, final embeddings, and embedding costs keyed by momentum.
    - process:
        - For each EE momentum, run EE stage, then run fixed embedding stage from EE endpoint.
    """
    ee_states = {}
    final_embeddings = {}
    embedding_costs = {}

    # Shared embedding runner. We reuse it by resetting its buffers for each variant.
    embedding_runner = utils.original_tSNE(
        labels=args.labels,
        sample_data="mnist",
        perplexity=args.perplexity,
        sample_number=args.sample_number,
        total_iteration=args.total_iterations,
        ee_iteration=args.ee_iterations,
        alpha=args.alpha,
        learning_rate=args.embedding_learning_rate,
        initial_momentum=args.embedding_initial_momentum,
        final_momentum=args.embedding_final_momentum,
        momentum_switch_iteration=args.embedding_momentum_switch_iteration,
        pca_components=args.pca_components,
        initial_beta_coefficient=args.initial_beta_coefficient,
        tol=args.tol,
        random_state=args.seed,
    )
    embedding_runner.X_sampled = base.X_sampled.copy()
    embedding_runner.y_sampled = np.array(base.y_sampled).copy()
    embedding_runner.N = base.N
    embedding_runner.P = base.P.copy()

    for momentum in args.ee_momentums:
        key = float(momentum)
        label = momentum_label(key)

        # Use MM with a constant momentum coefficient in the EE stage.
        # For momentum=0.0, MM is equivalent to GD.
        ee_runner = utils.original_tSNE(
            labels=args.labels,
            sample_data="mnist",
            perplexity=args.perplexity,
            sample_number=args.sample_number,
            total_iteration=args.ee_iterations,
            ee_iteration=args.ee_iterations,
            alpha=args.alpha,
            learning_rate=args.ee_learning_rate,
            initial_momentum=key,
            final_momentum=key,
            momentum_switch_iteration=args.ee_iterations + 1,
            pca_components=args.pca_components,
            initial_beta_coefficient=args.initial_beta_coefficient,
            tol=args.tol,
            random_state=args.seed,
        )
        ee_runner.X_sampled = base.X_sampled.copy()
        ee_runner.y_sampled = np.array(base.y_sampled).copy()
        ee_runner.N = base.N
        ee_runner.P = base.P.copy()

        ee_label = (
            f"proc=discrete | method=MM | ee_momentum={key:.1f} "
            f"| ee_iter={args.ee_iterations}"
        )
        print(f"[{ee_label}] stage start: early_exaggeration (0 to {args.ee_iterations - 1})")
        ee_runner._log_prefix = ee_label
        ee_runner._log_stage = "early_exaggeration"
        ee_runner._log_interval = args.log_interval

        ee_runner.fit_transform(
            X=ee_runner.X_sampled,
            store_history=True,
            verbose=True,
            seed=args.seed,
            optimization_method="MM",
            initial_Y=initial_Y,
        )

        Y_ee = ee_runner.Y[args.ee_iterations].copy()
        Y_prev = ee_runner.Y[args.ee_iterations - 1].copy()
        ee_cost = np.array(ee_runner.kl_history)
        print(f"[{ee_label}] stage end")

        ee_states[key] = {
            "label": label,
            "Y_ee": Y_ee,
            "Y_prev": Y_prev,
            "ee_cost": ee_cost,
            "ee_iteration": args.ee_iterations,
            "ee_momentum": key,
        }

        embed_label = (
            f"proc=discrete+embedding | method=MM "
            f"| ee_momentum={key:.1f} | ee_iter={args.ee_iterations}"
        )
        final_Y, emb_cost = _run_embedding_stage_from_state(
            embedding_runner=embedding_runner,
            args=args,
            Y_ee=Y_ee,
            log_prefix=embed_label,
            ee_iteration=args.ee_iterations,
        )

        final_embeddings[key] = final_Y
        embedding_costs[key] = emb_cost

    return ee_states, final_embeddings, embedding_costs


def build_summary_dataframe(ee_states, final_embeddings, embedding_costs):
    """
    Build per-momentum summary table for EE and final stages.

    - inputs:
        - EE states dict, final embeddings dict, and embedding costs dict.
    - outputs:
        - Summary DataFrame with costs and geometric statistics.
    - process:
        - Aggregate last costs and computed stats into one row per momentum.
    """
    rows = []
    for momentum in sorted(ee_states.keys()):
        ee_cost = ee_states[momentum]["ee_cost"]
        emb_cost = embedding_costs[momentum]
        ee_stats = _summary_stats(ee_states[momentum]["Y_ee"])
        final_stats = _summary_stats(final_embeddings[momentum])

        row = {
            "group": "discrete_mm_momentum_sweep",
            "method": "MM",
            "ee_momentum": float(momentum),
            "ee_momentum_label": ee_states[momentum]["label"],
            "ee_iteration": int(ee_states[momentum]["ee_iteration"]),
            "ee_cost_last": float(ee_cost[-1]) if len(ee_cost) else np.nan,
            "embedding_cost_last": float(emb_cost[-1]) if len(emb_cost) else np.nan,
        }
        row.update({f"ee_{k}": v for k, v in ee_stats.items()})
        row.update({f"final_{k}": v for k, v in final_stats.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def build_cost_dataframe(ee_states, embedding_costs):
    """
    Build long-format cost table across EE and embedding stages.

    - inputs:
        - EE states dict and embedding costs dict.
    - outputs:
        - DataFrame with stage, iteration, momentum, and cost values.
    - process:
        - Flatten per-stage cost arrays into row records for plotting/analysis.
    """
    rows = []
    for momentum in sorted(ee_states.keys()):
        label = ee_states[momentum]["label"]

        for local_i, cost in enumerate(ee_states[momentum]["ee_cost"]):
            rows.append({
                "group": "discrete_mm_momentum_sweep",
                "method": "MM",
                "ee_momentum": float(momentum),
                "ee_momentum_label": label,
                "stage": "early_exaggeration",
                "iteration": local_i,
                "cost": float(cost),
            })

        ee_iteration = ee_states[momentum]["ee_iteration"]
        for local_i, cost in enumerate(embedding_costs[momentum]):
            rows.append({
                "group": "discrete_mm_momentum_sweep",
                "method": "MM",
                "ee_momentum": float(momentum),
                "ee_momentum_label": label,
                "stage": "embedding",
                "iteration": ee_iteration + local_i,
                "cost": float(cost),
            })

    return pd.DataFrame(rows)


def save_outputs(args, base, initial_Y, ee_states, final_embeddings, embedding_costs):
    """
    Save sweep artifacts to npz and csv files.

    - inputs:
        - Args, base object, initial Y, EE states, final embeddings, and costs.
    - outputs:
        - Written npz/csv files and console paths.
    - process:
        - Create payloads/tables, write arrays and DataFrames, and print saved locations.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    npz_path = os.path.join(args.output_dir, "discrete_mm_momentum_sweep.npz")
    summary_csv = os.path.join(args.output_dir, "summary_discrete_mm_momentum_sweep.csv")
    costs_csv = os.path.join(args.output_dir, "costs_discrete_mm_momentum_sweep.csv")

    momentums = np.array(sorted(ee_states.keys()), dtype=float)

    savez_payload = {
        "labels": np.array(base.y_sampled),
        "initial_Y": initial_Y,
        "probabilities": base.P.copy(),
        "ee_momentums": momentums,
        "ee_iterations": np.array([args.ee_iterations]),
        "total_iterations": np.array([args.total_iterations]),
        "alpha": np.array([args.alpha]),
        "ee_learning_rate": np.array([args.ee_learning_rate]),
        "embedding_learning_rate": np.array([args.embedding_learning_rate]),
    }

    for momentum in momentums:
        key = float(momentum)
        label = ee_states[key]["label"]
        prefix = f"ee_momentum_{label}"

        savez_payload[f"{prefix}_state"] = ee_states[key]["Y_ee"]
        savez_payload[f"{prefix}_state_prev"] = ee_states[key]["Y_prev"]
        savez_payload[f"{prefix}_cost"] = ee_states[key]["ee_cost"]
        savez_payload[f"{prefix}_embedding_final"] = final_embeddings[key]
        savez_payload[f"{prefix}_embedding_cost"] = embedding_costs[key]

    np.savez(npz_path, **savez_payload)

    summary_df = build_summary_dataframe(ee_states, final_embeddings, embedding_costs)
    costs_df = build_cost_dataframe(ee_states, embedding_costs)

    summary_df.to_csv(summary_csv, index=False)
    costs_df.to_csv(costs_csv, index=False)

    print("\nSaved:")
    print(" ", npz_path)
    print(" ", summary_csv)
    print(" ", costs_csv)


def main():
    """
    Run the full discrete momentum sweep workflow.

    - inputs:
        - CLI arguments from the current process.
    - outputs:
        - Persisted sweep outputs under the target output directory.
    - process:
        - Parse args, build base data, generate initial Y, run variants, and save outputs.
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    base = build_data_and_probabilities(args)

    rng = np.random.RandomState(args.seed)
    initial_Y = rng.normal(
        loc=0.0,
        scale=base.initialization_std,
        size=(base.N, 2),
    )

    ee_states, final_embeddings, embedding_costs = run_all_variants(
        args=args,
        base=base,
        initial_Y=initial_Y,
    )

    save_outputs(
        args=args,
        base=base,
        initial_Y=initial_Y,
        ee_states=ee_states,
        final_embeddings=final_embeddings,
        embedding_costs=embedding_costs,
    )


if __name__ == "__main__":
    main()
