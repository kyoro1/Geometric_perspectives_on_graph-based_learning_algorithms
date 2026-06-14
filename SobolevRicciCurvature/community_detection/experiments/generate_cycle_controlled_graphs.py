#!/usr/bin/env python3
"""
Generate a one-parameter family of cycle-controlled graphs from a fixed point cloud.

Construction:
  1. Generate or load points X.
  2. Build a symmetric kNN candidate graph G_knn.
  3. Extract a minimum spanning tree T from G_knn.
  4. Add an eta-fraction of non-tree kNN edges to T.

If T is connected, each added non-tree edge increases the cycle rank beta_1 by one:
  beta_1(G_eta) = |E(G_eta)| - |V| + 1.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.neighbors import NearestNeighbors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cycle-controlled graphs using MST + kNN chord addition."
    )
    parser.add_argument(
        "--dataset",
        choices=["grid", "uniform", "two_moons", "swiss_roll"],
        default="grid",
    )
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--etas",
        nargs="+",
        type=float,
        default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0],
        help="Fractions of non-tree kNN edges to add.",
    )
    parser.add_argument(
        "--add-mode",
        choices=["short", "long", "random"],
        default="short",
        help="How to choose non-tree edges to add.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/cycle_controlled_graphs")
    return parser.parse_args()


def generate_points(args):
    rng = np.random.default_rng(args.seed)

    if args.dataset == "grid":
        g = int(args.grid_size)
        xs, ys = np.meshgrid(np.linspace(0.0, 1.0, g), np.linspace(0.0, 1.0, g))
        X = np.column_stack([xs.ravel(), ys.ravel()])
        if args.noise > 0:
            X = X + rng.normal(scale=args.noise, size=X.shape)
        if args.n < X.shape[0]:
            idx = rng.choice(X.shape[0], size=args.n, replace=False)
            X = X[idx]
        labels = np.zeros(X.shape[0], dtype=int)

    elif args.dataset == "uniform":
        X = rng.uniform(0.0, 1.0, size=(args.n, 2))
        labels = np.zeros(args.n, dtype=int)

    elif args.dataset == "two_moons":
        X, labels = make_moons(n_samples=args.n, noise=args.noise, random_state=args.seed)

    elif args.dataset == "swiss_roll":
        X3, t = make_swiss_roll(n_samples=args.n, noise=args.noise, random_state=args.seed)
        # Use intrinsic-looking 2D coordinates for plotting, but distances are computed in 3D if desired.
        # For simplicity here, use x-z coordinates to keep graph generation transparent.
        X = X3[:, [0, 2]]
        labels = pd.qcut(t, q=4, labels=False, duplicates="drop").astype(int)

    else:
        raise ValueError(args.dataset)

    # Compact 0..n-1 indexing.
    return np.asarray(X, dtype=float), np.asarray(labels, dtype=int)


def build_symmetric_knn_graph(X: np.ndarray, k: int) -> nx.Graph:
    n = X.shape[0]
    if k >= n:
        raise ValueError(f"k must be < n, got k={k}, n={n}")

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    G = nx.Graph()
    for i in range(n):
        G.add_node(int(i), x=float(X[i, 0]), y=float(X[i, 1]))

    for i in range(n):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            u, v = int(i), int(j)
            length = float(dist)
            if G.has_edge(u, v):
                if length < G[u][v]["length"]:
                    G[u][v]["length"] = length
                    G[u][v]["weight"] = length
            else:
                G.add_edge(u, v, length=length, weight=length)

    if not nx.is_connected(G):
        # kNN can be disconnected for small k. Connect components by adding nearest inter-component edges.
        components = [list(c) for c in nx.connected_components(G)]
        while len(components) > 1:
            c0 = components[0]
            best = None
            for a in c0:
                for comp_id in range(1, len(components)):
                    for b in components[comp_id]:
                        d = float(np.linalg.norm(X[a] - X[b]))
                        if best is None or d < best[0]:
                            best = (d, a, b, comp_id)
            d, a, b, comp_id = best
            G.add_edge(int(a), int(b), length=d, weight=d)
            components = [list(c) for c in nx.connected_components(G)]

    return G


def edge_key(u, v):
    u, v = int(u), int(v)
    return (u, v) if u <= v else (v, u)


def ordered_extra_edges(G_knn: nx.Graph, T: nx.Graph, mode: str, seed: int):
    tree_edges = {edge_key(u, v) for u, v in T.edges()}
    extra = []
    for u, v, d in G_knn.edges(data=True):
        e = edge_key(u, v)
        if e not in tree_edges:
            extra.append((e[0], e[1], float(d["length"])))

    if mode == "short":
        extra.sort(key=lambda x: x[2])
    elif mode == "long":
        extra.sort(key=lambda x: x[2], reverse=True)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(extra)
    else:
        raise ValueError(mode)

    return extra


def graph_from_tree_plus_edges(T: nx.Graph, extra_edges, m_add: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(T.nodes(data=True))

    for u, v, d in T.edges(data=True):
        length = float(d["length"])
        G.add_edge(int(u), int(v), length=length, weight=length, is_tree=1, is_extra=0)

    for u, v, length in extra_edges[:m_add]:
        G.add_edge(int(u), int(v), length=float(length), weight=float(length), is_tree=0, is_extra=1)

    return G


def save_graphml_int_safe(G: nx.Graph, path: Path):
    # GraphML likes simple scalar attrs.
    H = nx.Graph()
    for n, d in G.nodes(data=True):
        H.add_node(int(n), **{k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
                             for k, v in d.items()})
    for u, v, d in G.edges(data=True):
        H.add_edge(int(u), int(v), **{k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v)
                                      for k, v in d.items()})
    nx.write_graphml(H, path)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    graph_dir = out_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    X, labels = generate_points(args)
    n = X.shape[0]

    G_knn = build_symmetric_knn_graph(X, args.k)
    T = nx.minimum_spanning_tree(G_knn, weight="length")
    extra = ordered_extra_edges(G_knn, T, args.add_mode, args.seed)

    points_df = pd.DataFrame({
        "node": np.arange(n, dtype=int),
        "x": X[:, 0],
        "y": X[:, 1],
        "label": labels,
    })
    points_df.to_csv(out_dir / "points.csv", index=False)

    manifest_rows = []
    for eta in args.etas:
        eta = float(eta)
        eta_clipped = min(max(eta, 0.0), 1.0)
        m_add = int(np.floor(eta_clipped * len(extra)))
        G_eta = graph_from_tree_plus_edges(T, extra, m_add)

        beta1 = G_eta.number_of_edges() - G_eta.number_of_nodes() + nx.number_connected_components(G_eta)
        eta_actual = m_add / max(len(extra), 1)

        tag = f"eta_{eta_clipped:.4f}".replace(".", "p")
        graph_path = graph_dir / f"graph_{tag}.graphml"
        save_graphml_int_safe(G_eta, graph_path)

        manifest_rows.append({
            "eta_requested": eta,
            "eta_actual": eta_actual,
            "m_add": m_add,
            "n_extra_candidates": len(extra),
            "n_nodes": G_eta.number_of_nodes(),
            "n_edges": G_eta.number_of_edges(),
            "beta1": beta1,
            "k": args.k,
            "dataset": args.dataset,
            "add_mode": args.add_mode,
            "graph_path": str(graph_path.relative_to(out_dir)),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    metadata = {
        "dataset": args.dataset,
        "n": int(n),
        "k": int(args.k),
        "add_mode": args.add_mode,
        "seed": int(args.seed),
        "noise": float(args.noise),
        "n_knn_edges": int(G_knn.number_of_edges()),
        "n_tree_edges": int(T.number_of_edges()),
        "n_extra_candidates": int(len(extra)),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved:")
    print(" ", out_dir / "points.csv")
    print(" ", out_dir / "manifest.csv")
    print(" ", out_dir / "metadata.json")
    print(" ", graph_dir)


if __name__ == "__main__":
    main()
