import shutil
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance

import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib as mpl
mpl.rcParams["text.usetex"] = False
from typing import Dict, Any, Iterable

class Visualization():
    def __init__(self):
        pass

    def class_indicator_matrix(self,
                               y):
        """One-hot: Encode class labels into a binary matrix"""
        y = np.asarray(y)
        classes = np.unique(y)
        G = np.zeros((y.size, classes.size), dtype=float)
        for c_idx, c in enumerate(classes):
            G[:, c_idx] = (y == c).astype(float)
        return G
    
    def plot_pruning_subplots(self,
                              df: pd.DataFrame, 
                              ncols: int = 5, 
                              figsize: tuple = (20,5), 
                              highlight_label: str = "Distance",
                              output_file: str = "./results/pruning_performance.pdf",
                              title: str = None,
                              sup_title_fontsize: int = 24):
        """
        Display multiple datasets in one figure with subplots.

        Parameters:
            df: DataFrame containing the results with columns:
                - dataset
                - method
                - metric (either "good removed (%)" or "shortcut removed (%)")
                - mean
                - std
            ncols: Number of columns in the subplot grid.
            figsize: Size of the entire figure.
            highlight_label: Method label to highlight with a different color.
            output_file: Path to save the output figure.
        """
        if shutil.which("latex") is None:
            mpl.rcParams.update({
                "text.usetex": False,
                "font.family": "DejaVu Sans",
            })

        datasets = df["dataset"].unique()
        n = len(datasets)
        nrows = int(np.ceil(n / ncols))

        _, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Colors
        good_color   = 'skyblue'
        short_color  = 'salmon'
        # emphasize Color on Distance
        good_hi_color  = 'cornflowerblue'
        short_hi_color = 'tomato'

        for i, ds in enumerate(datasets):
            ax = axes[i]
            d = df[df["dataset"] == ds]
            mean_p = d.pivot_table(index="method", columns="metric", values="mean")
            std_p  = d.pivot_table(index="method", columns="metric", values="std")

            methods = mean_p.index.tolist()
            x = np.arange(len(methods))
            width = 0.4

            good_mean  = mean_p["good removed (%)"].to_numpy()
            good_std   = std_p["good removed (%)"].to_numpy()
            short_mean = mean_p["shortcut removed (%)"].to_numpy()
            short_std  = std_p["shortcut removed (%)"].to_numpy()

            # Highlight specific method
            is_dist = [m == highlight_label for m in methods]
            good_colors  = [good_hi_color if flag else good_color  for flag in is_dist]
            short_colors = [short_hi_color if flag else short_color for flag in is_dist]

            ax.bar(x - width/2, good_mean,  width, yerr=good_std,  capsize=3,
                label="good removed (%)",     color=good_colors,  edgecolor='white', linewidth=0.5)
            ax.bar(x + width/2, short_mean, width, yerr=short_std, capsize=3,
                label="shortcut removed (%)", color=short_colors, edgecolor='white', linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=14)
            ax.set_ylim(0, 100)
            ax.set_title(ds, fontsize=20)

            if i % ncols == 0:
                ax.set_ylabel("Percentage")
            else:
                ax.set_yticklabels([])

            if i == 0:
                ax.legend(fontsize=13)
            else:
                ax.legend().remove()

        for j in range(i+1, len(axes)):
            axes[j].axis("off")

        if title is not None:
            plt.suptitle(title, fontsize=sup_title_fontsize)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()

    def _traces_for_one_graph_2d(self,
        X, graph, shortcut_edges=None,
        edge_width=0.5, edge_color='lightgrey',
        shortcut_color='red', shortcut_width=None,
        node_color='#1f78b4', node_size=3,
        pad_ratio=0.05, axes=False
    ):
        """
        Generate Plotly traces for a single graph in 2D.

        Parameters:
            X: Node coordinates (n, 2)
            graph: NetworkX graph object
            shortcut_edges: List of edges to highlight as shortcuts (list of (u,v) tuples)
            edge_width: Width of normal edges
            edge_color: Color of normal edges
            shortcut_color: Color of shortcut edges
            shortcut_width: Width of shortcut edges (if None, set to max(edge_width, 1.0))
            node_color: Color of nodes
            node_size: Size of nodes
            pad_ratio: Padding ratio for axis limits
            axes: Whether to show axes
        Returns:
            edge_trace: Scattergl trace for normal edges
            sc_trace: Scattergl trace for shortcut edges
            node_trace: Scattergl trace for nodes
            axes_config: Dictionary with axis configuration
        """
        shortcut_set = set() if shortcut_edges is None else {
            tuple(sorted(map(int, e))) for e in shortcut_edges
        }

        base_x, base_y = [], []
        sc_x, sc_y = [], []
        for u, v in graph.edges():
            x0, y0 = X[u]
            x1, y1 = X[v]
            bx, by = (sc_x, sc_y) if tuple(sorted((u, v))) in shortcut_set else (base_x, base_y)
            bx.extend([x0, x1, None])
            by.extend([y0, y1, None])

        edge_trace = go.Scattergl(
            x=base_x, y=base_y, mode='lines',
            line=dict(width=edge_width, color=edge_color),
            hoverinfo='skip', showlegend=False
        )
        sc_trace = go.Scattergl(
            x=sc_x, y=sc_y, mode='lines',
            line=dict(width=shortcut_width or max(edge_width, 1.0), color=shortcut_color),
            hoverinfo='skip', showlegend=False, name='shortcut'
        )
        node_trace = go.Scattergl(
            x=X[:, 0], y=X[:, 1], mode='markers',
            marker=dict(size=node_size, color=node_color, opacity=0.9),
            showlegend=False
        )

        (xmin, ymin), (xmax, ymax) = X.min(axis=0), X.max(axis=0)
        dx, dy = xmax - xmin, ymax - ymin
        px, py = dx * pad_ratio, dy * pad_ratio
        axes_config = dict(
            x_range=[xmin - px, xmax + px],
            y_range=[ymin - py, ymax + py],
            axes_visible=axes
        )
        return edge_trace, sc_trace, node_trace, axes_config

    def plot_graphs_2D_side_by_side(self,
        X_list, G_list, shortcut_edges_list, titles=None,
        # Common style
        node_color='#1f78b4', node_size=3,
        edge_width=0.5, edge_color='lightgrey',
        shortcut_color='red', shortcut_width=None,
        pad_ratio=0.05, axes=False,
        # Layout/Font (Common)
        width=2200, height=480, margin=dict(l=10, r=10, t=60, b=10),
        suptitle=None, title_font_size=28, subtitle_font_size=22,
        font_family="Hiragino Sans, Noto Sans CJK JP, Arial, sans-serif",
        base_font_size=14, axis_title_size=14, axis_tick_size=12
    ):
        """
        Display multiple datasets in one figure with subplots (Plotly).

        Parameters:
            X_list: List of node coordinates for each dataset (each of shape (n, 2))
            G_list: List of NetworkX graph objects for each dataset
            shortcut_edges_list: List of lists of edges to highlight as shortcuts for each dataset
            titles: List of titles for each subplot. If None, defaults to "Dataset 1", "Dataset 2", ...
            node_color: Color of nodes
            node_size: Size of nodes
            edge_width: Width of normal edges
            edge_color: Color of normal edges
            shortcut_color: Color of shortcut edges
            shortcut_width: Width of shortcut edges (if None, set to max(edge_width, 1.0))
            pad_ratio: Padding ratio for axis limits
            axes: Whether to show axes
            width: Width of the entire figure
            height: Height of the entire figure
            margin: Margin dictionary for the figure layout
            suptitle: Overall title for the figure
            title_font_size: Font size for the overall title
            subtitle_font_size: Font size for subplot titles
            font_family: Font family for all text
            base_font_size: Base font size for all text
            axis_title_size: Font size for axis titles
            axis_tick_size: Font size for axis tick labels
        Returns:
            fig: Plotly Figure object containing the subplots
        """
        n = len(X_list)
        assert n == 5, f"Received {n} datasets, but 5 are required."
        if titles is None:
            titles = [f"Dataset {i+1}" for i in range(n)]

        fig = make_subplots(
            rows=1, cols=n,
            specs=[[{'type': 'xy'}] * n],
            subplot_titles=titles,
            horizontal_spacing=0.02
        )

        for i in range(n):
            edge_t, sc_t, node_t, cfg = self._traces_for_one_graph_2d(
                X_list[i], G_list[i], shortcut_edges_list[i],
                edge_width=edge_width, edge_color=edge_color,
                shortcut_color=shortcut_color, shortcut_width=shortcut_width,
                node_color=node_color, node_size=node_size,
                pad_ratio=pad_ratio, axes=axes
            )
            fig.add_trace(edge_t, row=1, col=i+1)
            fig.add_trace(sc_t,   row=1, col=i+1)
            fig.add_trace(node_t, row=1, col=i+1)

            # Equal scale, range, and axis visibility
            xkey = 'xaxis' if i == 0 else f'xaxis{i+1}'
            ykey = 'yaxis' if i == 0 else f'yaxis{i+1}'
            fig.update_layout(**{
                xkey: dict(
                    range=cfg['x_range'],
                    scaleanchor=ykey.replace('axis', ''), scaleratio=1,
                    showgrid=False, zeroline=False, visible=axes,
                    title=dict(text='x', font=dict(size=axis_title_size)) if axes else None,
                    tickfont=dict(size=axis_tick_size) if axes else None
                ),
                ykey: dict(
                    range=cfg['y_range'],
                    showgrid=False, zeroline=False, visible=axes,
                    title=dict(text='y', font=dict(size=axis_title_size)) if axes else None,
                    tickfont=dict(size=axis_tick_size) if axes else None
                ),
            })

        # No background
        fig.update_layout(
            width=width, height=height, margin=margin,
            showlegend=False,
            font=dict(family=font_family, size=base_font_size),
            template=None, 
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        if suptitle:
            fig.update_layout(title=dict(text=suptitle, font=dict(size=title_font_size)))

        # Subtitle font
        if hasattr(fig.layout, "annotations"):
            for ann in fig.layout.annotations:
                if ann.text in titles:
                    ann.font = dict(family=font_family, size=subtitle_font_size)

        return fig
    
    def _traces_for_one_graph(self,
        X, graph, shortcut_edges=None,
        edge_width=0.5, edge_color='lightgrey',
        shortcut_color='red', shortcut_width=None,
        node_color='#1f78b4', node_size=3,
        cmap='Viridis', opacity=None, cmin=-1, cmax=1,
        node_colorbar=False, node_colorbar_title=None,
        pad_ratio=0.04, axes=False
    ):
        shortcut_set = set() if shortcut_edges is None else {
            tuple(sorted(map(int, e))) for e in shortcut_edges
        }

        base_x, base_y, base_z = [], [], []
        sc_x, sc_y, sc_z = [], [], []
        for u, v in graph.edges():
            x0, y0, z0 = X[u]
            x1, y1, z1 = X[v]
            bx, by, bz = (sc_x, sc_y, sc_z) if tuple(sorted((u, v))) in shortcut_set else (base_x, base_y, base_z)
            bx.extend([x0, x1, None])
            by.extend([y0, y1, None])
            bz.extend([z0, z1, None])

        edge_trace = go.Scatter3d(
            x=base_x, y=base_y, z=base_z,
            mode='lines',
            line=dict(width=edge_width, color=edge_color),
            opacity=opacity, showlegend=False, hoverinfo='skip',
        )
        sc_trace = go.Scatter3d(
            x=sc_x, y=sc_y, z=sc_z,
            mode='lines',
            line=dict(width=shortcut_width or max(edge_width, 1.0), color=shortcut_color),
            opacity=opacity, showlegend=False, hoverinfo='skip',
            name='shortcut'
        )
        node_trace = go.Scatter3d(
            x=X[:,0], y=X[:,1], z=X[:,2],
            mode='markers',
            marker=dict(
                size=node_size, color=node_color,
                colorscale=cmap, opacity=0.8, cmin=cmin, cmax=cmax,
                colorbar=dict(
                    title=node_colorbar_title, thickness=40,
                    xanchor='left', titleside='right', tickfont=dict(size=30),
                ) if node_colorbar else None,
            ),
            showlegend=False
        )

        x_min, y_min, z_min = X.min(axis=0)
        x_max, y_max, z_max = X.max(axis=0)
        dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
        px, py, pz = dx*pad_ratio, dy*pad_ratio, dz*pad_ratio
        scene_axes = dict(
            xaxis=dict(range=[x_min - px, x_max + px], visible=not axes if axes else False),
            yaxis=dict(range=[y_min - py, y_max + py], visible=not axes if axes else False),
            zaxis=dict(range=[z_min - pz, z_max + pz], visible=not axes if axes else False),
            aspectmode='data'
        )
        return edge_trace, sc_trace, node_trace, scene_axes

    def plot_graphs_3D_side_by_side(self,
        X_list, G_list, shortcut_edges_list, titles=None,
        # Common style
        node_color='#1f78b4', node_size=3,
        edge_width=0.5, edge_color='lightgrey',
        shortcut_color='red', shortcut_width=None,
        cmap='Viridis', opacity=None, cmin=-1, cmax=1,
        axes=False, pad_ratio=0.05,
        width=2200, height=650, margin=dict(l=0, r=0, t=60, b=0),
        camera=None,
        # Layout/Font
        suptitle=None,                 # Title for the entire figure
        title_font_size=30,            # Font size for the entire figure title
        subtitle_font_size=24,         # Font size for each subplot title (e.g., torii)
        font_family="Hiragino Sans, Noto Sans CJK JP, sans-serif",
        base_font_size=14,             # Default font size for the entire figure
        axis_title_size=14,
        axis_tick_size=12
    ):
        """
        Display multiple datasets in one figure with subplots (Plotly 3D).

        Parameters:
            X_list: List of node coordinates for each dataset (each of shape (n, 3))
            G_list: List of NetworkX graph objects for each dataset
            shortcut_edges_list: List of lists of edges to highlight as shortcuts for each dataset
            titles: List of titles for each subplot. If None, defaults to "Dataset 1", "Dataset 2", ...
            node_color: Color of nodes
            node_size: Size of nodes
            edge_width: Width of normal edges
            edge_color: Color of normal edges
            shortcut_color: Color of shortcut edges
            shortcut_width: Width of shortcut edges (if None, set to max(edge_width, 1.0))
            cmap: Colormap for node colors
            opacity: Opacity for edges and nodes
            cmin: Minimum value for colormap scaling
            cmax: Maximum value for colormap scaling
            axes: Whether to show axes
            pad_ratio: Padding ratio for axis limits
            width: Width of the entire figure
            height: Height of the entire figure
            margin: Margin dictionary for the figure layout
            camera: Camera configuration dictionary for 3D scenes
            suptitle: Overall title for the figure
            title_font_size: Font size for the overall title
            subtitle_font_size: Font size for subplot titles
            font_family: Font family for all text
            base_font_size: Base font size for all text
            axis_title_size: Font size for axis titles
            axis_tick_size: Font size for axis tick labels
        Returns:
            fig: Plotly Figure object containing the subplots        
        """
        n = len(X_list)
        assert n == 5, f"Received unexpected number of datasets: {n}, but 5 are required."

        if titles is None:
            titles = [f"Dataset {i+1}" for i in range(n)]

        fig = make_subplots(
            rows=1, cols=n,
            specs=[[{'type': 'scene'}]*n],
            subplot_titles=titles,
            horizontal_spacing=0.02
        )

        for i in range(n):
            edge_t, sc_t, node_t, scene_axes = self._traces_for_one_graph(
                X_list[i], G_list[i], shortcut_edges_list[i],
                edge_width=edge_width, edge_color=edge_color,
                shortcut_color=shortcut_color, shortcut_width=shortcut_width,
                node_color=node_color, node_size=node_size,
                cmap=cmap, opacity=opacity, cmin=cmin, cmax=cmax,
                pad_ratio=pad_ratio, axes=axes
            )
            fig.add_trace(edge_t, row=1, col=i+1)
            fig.add_trace(sc_t,   row=1, col=i+1)
            fig.add_trace(node_t, row=1, col=i+1)

            scene_key = f"scene{i+1}" if i > 0 else "scene"
            # axis title, tick font size
            scene_axes.update(dict(
                xaxis=dict(**scene_axes["xaxis"], title=dict(text="x", font=dict(size=axis_title_size)),
                        tickfont=dict(size=axis_tick_size)),
                yaxis=dict(**scene_axes["yaxis"], title=dict(text="y", font=dict(size=axis_title_size)),
                        tickfont=dict(size=axis_tick_size)),
                zaxis=dict(**scene_axes["zaxis"], title=dict(text="z", font=dict(size=axis_title_size)),
                        tickfont=dict(size=axis_tick_size)),
            ))
            fig.update_layout(**{scene_key: scene_axes})
            if camera is not None:
                fig.update_layout(**{scene_key+"_camera": camera})

        # Configure layout
        fig.update_layout(
            width=width, height=height, margin=margin,
            showlegend=False,
            font=dict(family=font_family, size=base_font_size)
        )

        # Overall title
        if suptitle:
            fig.update_layout(title=dict(text=suptitle, font=dict(size=title_font_size)))

        # Subplot titles (annotations) font size adjustment
        if hasattr(fig.layout, "annotations"):
            for ann in fig.layout.annotations:
                if ann.text in titles:  # Subplot titles only
                    ann.font = dict(family=font_family, size=subtitle_font_size)
        return fig
    
    @staticmethod
    def plot_distribution_orc_src(orc: np.ndarray,
                                  src: np.ndarray,
                                  output_file: str,):
        plt.figure(figsize=(8,6))
        sns.histplot(orc, kde=False, label="ORC", bins=80)
        sns.histplot(src, kde=False, label="SRC(SPT)", bins=120)
        plt.legend(fontsize=15)
        plt.xlim(-1.0, 1.0)
        plt.xlabel("Curvature values", fontsize=15)
        plt.savefig(output_file)

    @staticmethod
    def plot_time_comparison(df_all: pd.DataFrame):
        color_map = {
            "Distance": "tab:blue",
            "ORC only": "tab:blue",
            "ORC-MANL": "tab:blue",
            "SRC only": "tab:orange",
            "SRC-MANL": "tab:orange",
        }

        df_plot = (
            df_all
            .groupby(["dataset", "method"], as_index=False)["time_total"]
            .mean()
        )

        dataset_order = [
            "concentric_circles", "moons", "s_curve", "cassini", "mixture_of_gaussians",
            "torii", "hyperboloids", "parab_and_hyp", "double_paraboloid", "3D_swiss_roll"
        ]

        fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
        axes = axes.flatten()

        method_order = ["Distance", "ORC only", "ORC-MANL", "SRC only", "SRC-MANL"]

        for ax, ds in zip(axes, dataset_order):
            df_ds = df_plot[df_plot["dataset"] == ds]
            df_ds = df_ds.set_index("method").loc[method_order].reset_index()

            colors = [color_map[m] for m in df_ds["method"]]

            ax.bar(df_ds["method"], df_ds["time_total"], color=colors)
            ax.set_title(ds)
            ax.set_xticklabels(df_ds["method"], rotation=30)

            if ax in axes[::5]:
                ax.set_ylabel("time_total (sec)")

        for i in range(len(dataset_order), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig("./results/time_comparison.pdf")
        plt.show()