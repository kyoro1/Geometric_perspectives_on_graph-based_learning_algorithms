import argparse
import os

import numpy as np
import pandas as pd

import utils


OPTIMIZATION_METHODS = ("GD", "MM", "NAG")
EE_GROUPS = ("discrete", "continuous_h400")


def parse_args():
    """
    Parse command-line options for h=400 comparison computation.

    - inputs:
        - CLI arguments from the current process.
    - outputs:
        - argparse.Namespace with EE/embedding and output configuration.
    - process:
        - Define parser flags for discrete/continuous settings and parse args.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute 6 EE variants (discrete + continuous@h400) and run the same "
            "original t-SNE embedding stage. This script writes npz/csv outputs only; "
            "use plot_embedding_stage_gd_mm_nag_h400_only.py for visualization."
        )
    )
    parser.add_argument("--labels", nargs="+", default=["2", "4", "6", "8"])
    parser.add_argument("--sample-number", type=int, default=400)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--ee-iterations", type=int, default=50)
    parser.add_argument("--total-iterations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=4.0)

    parser.add_argument("--discrete-learning-rate", type=float, default=100.0)
    parser.add_argument("--discrete-initial-momentum", type=float, default=0.5)
    parser.add_argument("--discrete-final-momentum", type=float, default=0.5)
    parser.add_argument("--discrete-momentum-switch-iteration", type=int, default=5)

    # Backward-compatible option. If two values such as "400 200" are passed,
    # this script uses the last value only, i.e. 400, and uses h=400.
    parser.add_argument("--continuous-h-values", nargs="+", type=float, default=[400.0])
    parser.add_argument(
        "--continuous-h-value",
        type=float,
        default=None,
        help=(
            "Continuous EE h value to use. If omitted, the first value of "
            "--continuous-h-values is used. For this script, pass "
            "--continuous-h-values 400 or --continuous-h-value 400."
        ),
    )
    parser.add_argument("--continuous-momentum", type=float, default=0.5)
    parser.add_argument(
        "--continuous-h400-ee-iterations",
        nargs=3,
        type=int,
        metavar=("GD", "MM", "NAG"),
        default=None,
        help=(
            "Method-specific EE iterations for continuous h=400. "
            "Example: --continuous-h400-ee-iterations 7 4 14"
        ),
    )

    parser.add_argument("--embedding-learning-rate", type=float, default=100.0)
    parser.add_argument("--embedding-initial-momentum", type=float, default=0.5)
    parser.add_argument("--embedding-final-momentum", type=float, default=0.8)
    parser.add_argument("--embedding-momentum-switch-iteration", type=int, default=250)

    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--initial-beta-coefficient", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/comparison_embedding_stage_gd_mm_nag_h400_only")
    return parser.parse_args()


def resolve_continuous_h_value(args):
    """
    Resolve the continuous h value from backward-compatible options.

    - inputs:
        - Parsed args containing continuous-h options.
    - outputs:
        - Float h value used for continuous EE runs.
    - process:
        - Prefer explicit single value, then fallback to list first element, then default.
    """
    if args.continuous_h_value is not None:
        return float(args.continuous_h_value)
    if args.continuous_h_values is None or len(args.continuous_h_values) == 0:
        return 400.0
    # For this h=400 script, use the first supplied value.
    # Example: --continuous-h-values 400
    return float(args.continuous_h_values[0])


def build_data_and_probabilities(args):
    """
    Prepare base dataset and t-SNE probability matrix.

    - inputs:
        - Parsed args with data and optimization settings.
    - outputs:
        - Initialized original_tSNE object with sampled data and P.
    - process:
        - Instantiate runner, load MNIST subset, and compute t-SNE probabilities.
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


def make_solution_path(args, base, initial_Y, h_value):
    """
    Construct a SolutionPath instance for continuous EE dynamics.

    - inputs:
        - Parsed args, prepared base object, initial Y, and h value.
    - outputs:
        - Ready-to-use SolutionPath object with initialized buffers/state.
    - process:
        - Configure SolutionPath parameters, copy data/P/initial state, and precompute setup.
    """
    solution_path = utils.SolutionPath(
        labels=args.labels,
        sample_data="mnist",
        perplexity=args.perplexity,
        alpha=args.alpha,
        h=h_value,
        total_iteration=args.ee_iterations,
        momentum=args.continuous_momentum,
        sample_number=args.sample_number,
        R=len(args.labels),
        m=1,
    )
    solution_path.X_sampled = base.X_sampled.copy()
    solution_path.y_sampled = np.array(base.y_sampled).copy()
    solution_path.N = base.X_sampled.shape[0]
    solution_path.num_unique_labels = np.unique(solution_path.y_sampled).shape[0]
    solution_path.P = base.P.copy()
    solution_path.Y = {0: initial_Y.copy(), -1: initial_Y.copy()}
    solution_path.Y_nes = {0: initial_Y.copy()}
    solution_path.Q = dict()
    solution_path.S = dict()
    solution_path.cost = dict()
    solution_path.squared_distances = dict()
    solution_path.student_kernel = dict()
    solution_path.getSolution()
    return solution_path


def get_continuous_ee_iteration(args, method):
    """
    Get method-specific EE iteration count for continuous runs.

    - inputs:
        - Parsed args and optimization method name.
    - outputs:
        - Integer EE iteration for the method.
    - process:
        - Use explicit per-method override if provided, otherwise use common EE iterations.
    """
    if args.continuous_h400_ee_iterations is not None:
        method_to_iter = dict(zip(OPTIMIZATION_METHODS, args.continuous_h400_ee_iterations))
        return method_to_iter[method]
    return args.ee_iterations


def _run_embedding_stage_from_state(
    embedding_runner,
    args,
    Y_ee,
    Y_prev,
    log_prefix,
    ee_iteration,
):
    """
    Run the common embedding stage from a given EE end state.

    - inputs:
        - Embedding runner, args, EE endpoint states, log prefix, and EE iteration index.
    - outputs:
        - Final embedding array and embedding-stage cost trajectory.
    - process:
        - Reset stage buffers/velocity at EE boundary, execute MM embedding updates,
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
        - Center points, compute covariance eigenvalues, and derive summary metrics.
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
    Execute all six EE variants and their common embedding continuation.

    - inputs:
        - Parsed args, prepared base object, and shared initial Y.
    - outputs:
        - EE states dict, final embeddings dict, and embedding costs dict.
    - process:
        - Run discrete variants, run continuous h=400 variants, then continue each with
          the same embedding-stage procedure.
    """
    final_embeddings = {}
    embedding_costs = {}
    ee_states = {}
    continuous_h_value = resolve_continuous_h_value(args)

    print(f"Using continuous h={continuous_h_value:g}; only h=400 continuous variants are computed.")

    # --- shared embedding runner ---
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

    # --- discrete variants ---
    for method in OPTIMIZATION_METHODS:
        discrete_runner = utils.original_tSNE(
            labels=args.labels,
            sample_data="mnist",
            perplexity=args.perplexity,
            sample_number=args.sample_number,
            total_iteration=args.ee_iterations,
            ee_iteration=args.ee_iterations,
            alpha=args.alpha,
            learning_rate=args.discrete_learning_rate,
            initial_momentum=args.discrete_initial_momentum,
            final_momentum=args.discrete_final_momentum,
            momentum_switch_iteration=args.discrete_momentum_switch_iteration,
            pca_components=args.pca_components,
            initial_beta_coefficient=args.initial_beta_coefficient,
            tol=args.tol,
            random_state=args.seed,
        )
        discrete_runner.X_sampled = base.X_sampled.copy()
        discrete_runner.y_sampled = np.array(base.y_sampled).copy()
        discrete_runner.N = base.N
        discrete_runner.P = base.P.copy()

        ee_label = f"proc=discrete | method={method} | ee_iter={args.ee_iterations}"
        print(f"[{ee_label}] stage start: early_exaggeration (0 to {args.ee_iterations - 1})")
        discrete_runner._log_prefix = ee_label
        discrete_runner._log_stage = "early_exaggeration"
        discrete_runner._log_interval = args.log_interval
        discrete_runner.fit_transform(
            X=discrete_runner.X_sampled,
            store_history=True,
            verbose=True,
            seed=args.seed,
            optimization_method=method,
            initial_Y=initial_Y,
        )
        Y_ee = discrete_runner.Y[args.ee_iterations].copy()
        Y_prev = discrete_runner.Y[args.ee_iterations - 1].copy()
        ee_cost = np.array(discrete_runner.kl_history)
        print(f"[{ee_label}] stage end")

        key = ("discrete", method)
        ee_states[key] = {
            "Y_ee": Y_ee,
            "Y_prev": Y_prev,
            "ee_cost": ee_cost,
            "ee_iteration": args.ee_iterations,
            "h": np.nan,
        }
        embed_label = f"proc=discrete+embedding | ee=discrete:{method} | ee_iter={args.ee_iterations}"
        final_Y, emb_cost = _run_embedding_stage_from_state(
            embedding_runner,
            args,
            Y_ee,
            Y_prev,
            embed_label,
            ee_iteration=args.ee_iterations,
        )
        final_embeddings[key] = final_Y
        embedding_costs[key] = emb_cost

    # --- continuous h=400 variants ---
    group_name = "continuous_h400"
    h_value = continuous_h_value
    solution_path = make_solution_path(args=args, base=base, initial_Y=initial_Y, h_value=h_value)

    for method in OPTIMIZATION_METHODS:
        ee_iteration = get_continuous_ee_iteration(args, method)
        ee_label = f"proc=continuous | h={h_value:g} | method={method} | ee_iter={ee_iteration}"
        print(f"[{ee_label}] stage start: early_exaggeration (0 to {ee_iteration - 1})")

        solution_path._log_prefix = ee_label
        solution_path._log_stage = "early_exaggeration"
        solution_path._log_interval = args.log_interval

        solution_path.ee_iteration = ee_iteration
        solution_path.Y = {0: initial_Y.copy(), -1: initial_Y.copy()}
        solution_path.Y_nes = {0: initial_Y.copy()}
        solution_path.Q = dict()
        solution_path.S = dict()
        solution_path.cost = dict()
        solution_path.squared_distances = dict()
        solution_path.student_kernel = dict()
        solution_path.getSolutionPath(optimization_method=method)

        Y_ee = solution_path.Y[ee_iteration].copy()
        Y_prev = solution_path.Y[ee_iteration - 1].copy()
        ee_cost = np.array([solution_path.cost[k] for k in range(ee_iteration)])
        print(f"[{ee_label}] stage end")

        key = (group_name, method)
        ee_states[key] = {
            "Y_ee": Y_ee,
            "Y_prev": Y_prev,
            "ee_cost": ee_cost,
            "ee_iteration": ee_iteration,
            "h": h_value,
        }

        embed_label = (
            f"proc=continuous+embedding | ee={group_name}:{method} "
            f"| h={h_value:g} | ee_iter={ee_iteration}"
        )
        final_Y, emb_cost = _run_embedding_stage_from_state(
            embedding_runner,
            args,
            Y_ee,
            Y_prev,
            embed_label,
            ee_iteration=ee_iteration,
        )
        final_embeddings[key] = final_Y
        embedding_costs[key] = emb_cost

    return ee_states, final_embeddings, embedding_costs


def build_summary_dataframe(ee_states, final_embeddings, embedding_costs):
    """
    Build summary table for EE and final embedding outcomes.

    - inputs:
        - EE states dict, final embeddings dict, and embedding costs dict.
    - outputs:
        - Summary DataFrame with per-group/method statistics.
    - process:
        - Aggregate final costs and geometric stats into one row per variant.
    """
    rows = []
    for group_name in EE_GROUPS:
        for method in OPTIMIZATION_METHODS:
            key = (group_name, method)
            ee_cost = ee_states[key]["ee_cost"]
            emb_cost = embedding_costs[key]
            ee_stats = _summary_stats(ee_states[key]["Y_ee"])
            final_stats = _summary_stats(final_embeddings[key])

            row = {
                "group": group_name,
                "method": method,
                "h": ee_states[key]["h"],
                "ee_iteration": ee_states[key]["ee_iteration"],
                "ee_cost_last": float(ee_cost[-1]) if len(ee_cost) else np.nan,
                "embedding_cost_last": float(emb_cost[-1]) if len(emb_cost) else np.nan,
            }
            row.update({f"ee_{k}": v for k, v in ee_stats.items()})
            row.update({f"final_{k}": v for k, v in final_stats.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def build_cost_dataframe(ee_states, embedding_costs):
    """
    Build long-format cost table across both optimization stages.

    - inputs:
        - EE states dict and embedding costs dict.
    - outputs:
        - DataFrame with group/method/stage/iteration/cost rows.
    - process:
        - Flatten EE and embedding cost trajectories into a single table.
    """
    rows = []
    for group_name in EE_GROUPS:
        for method in OPTIMIZATION_METHODS:
            key = (group_name, method)
            for local_i, cost in enumerate(ee_states[key]["ee_cost"]):
                rows.append({
                    "group": group_name,
                    "method": method,
                    "stage": "early_exaggeration",
                    "iteration": local_i,
                    "cost": float(cost),
                })
            ee_iteration = ee_states[key]["ee_iteration"]
            for local_i, cost in enumerate(embedding_costs[key]):
                rows.append({
                    "group": group_name,
                    "method": method,
                    "stage": "embedding",
                    "iteration": ee_iteration + local_i,
                    "cost": float(cost),
                })
    return pd.DataFrame(rows)


def print_difference_checks(ee_states, final_embeddings):
    """
    Print discrete vs continuous_h400 difference diagnostics.

    - inputs:
        - EE states dict and final embeddings dict.
    - outputs:
        - Console lines with absolute and relative norm differences.
    - process:
        - Compare matched method pairs at EE and final stages and report norms.
    """
    print("\\n=== Difference check: discrete vs continuous_h400 ===")
    for method in OPTIMIZATION_METHODS:
        key_discrete = ("discrete", method)
        key_cont = ("continuous_h400", method)

        ee_diff = np.linalg.norm(ee_states[key_discrete]["Y_ee"] - ee_states[key_cont]["Y_ee"])
        final_diff = np.linalg.norm(final_embeddings[key_discrete] - final_embeddings[key_cont])

        ee_rel = ee_diff / (np.linalg.norm(ee_states[key_discrete]["Y_ee"]) + 1e-12)
        final_rel = final_diff / (np.linalg.norm(final_embeddings[key_discrete]) + 1e-12)

        print(
            f"{method}: "
            f"EE diff={ee_diff:.6e} (rel={ee_rel:.6e}), "
            f"final diff={final_diff:.6e} (rel={final_rel:.6e})"
        )


def save_outputs(args, base, initial_Y, ee_states, final_embeddings, embedding_costs):
    """
    Save computed arrays and summaries for downstream plotting.

    - inputs:
        - Args, base object, initial Y, EE states, final embeddings, and costs.
    - outputs:
        - Written npz/csv files and printed save paths.
    - process:
        - Build savez payload and DataFrames, write files, and print output locations.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    continuous_h_value = resolve_continuous_h_value(args)

    npz_path = os.path.join(args.output_dir, "comparison_embedding_stage_gd_mm_nag_h400_only.npz")
    summary_csv = os.path.join(args.output_dir, "summary_h400_only.csv")
    costs_csv = os.path.join(args.output_dir, "costs_h400_only.csv")

    savez_payload = {
        "labels": np.array(base.y_sampled),
        "initial_Y": initial_Y,
        "probabilities": base.P.copy(),
        "ee_iterations": np.array([args.ee_iterations]),
        "total_iterations": np.array([args.total_iterations]),
        "continuous_h_value": np.array([continuous_h_value]),
        "continuous_h400_ee_iterations": (
            np.array(args.continuous_h400_ee_iterations)
            if args.continuous_h400_ee_iterations is not None
            else np.array([])
        ),
    }

    for group_name in EE_GROUPS:
        for method in OPTIMIZATION_METHODS:
            key = (group_name, method)
            prefix = f"ee_{group_name}_{method.lower()}"
            savez_payload[f"{prefix}_state"] = ee_states[key]["Y_ee"]
            savez_payload[f"{prefix}_state_prev"] = ee_states[key]["Y_prev"]
            savez_payload[f"{prefix}_cost"] = ee_states[key]["ee_cost"]
            savez_payload[f"{prefix}_ee_iteration"] = np.array([ee_states[key]["ee_iteration"]])
            savez_payload[f"{prefix}_h"] = np.array([ee_states[key]["h"]])
            savez_payload[f"{prefix}_embedding_final"] = final_embeddings[key]
            savez_payload[f"{prefix}_embedding_cost"] = embedding_costs[key]

    np.savez(npz_path, **savez_payload)

    summary_df = build_summary_dataframe(ee_states, final_embeddings, embedding_costs)
    costs_df = build_cost_dataframe(ee_states, embedding_costs)
    summary_df.to_csv(summary_csv, index=False)
    costs_df.to_csv(costs_csv, index=False)

    print("\\nSaved:")
    print(" ", npz_path)
    print(" ", summary_csv)
    print(" ", costs_csv)


def main():
    """
    Run the full h=400 comparison computation pipeline.

    - inputs:
        - CLI arguments from the current process.
    - outputs:
        - Persisted h=400 comparison artifacts under output directory.
    - process:
        - Parse args, prepare data/initial state, run variants, print checks, and save outputs.
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

    print_difference_checks(ee_states, final_embeddings)
    save_outputs(args, base, initial_Y, ee_states, final_embeddings, embedding_costs)


if __name__ == "__main__":
    main()
