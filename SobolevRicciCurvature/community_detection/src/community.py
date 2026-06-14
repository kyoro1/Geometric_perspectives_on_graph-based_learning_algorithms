import math
import numpy as np
import networkx as nx
import igraph as ig
import random

def add_similarity_weight_from_length(
    G: nx.Graph,
    length_attr: str = "length",
    weight_attr: str = "weight",
    beta: float = 1.0,
    eps: float = 1e-12,
    ) -> None:
    '''
    Add similarity weights to edges based on their lengths.
    - Inputs:
        - G: nx.Graph
        - length_attr: str
        - weight_attr: str
        - beta: float
        - eps: float
    - Outputs:
        - None (modifies G in place)
    - Proedure:
        For each edge, compute weight as w = exp(-beta * L), where L is the length attribute.
        If L is not finite, use L = 1.0.
        Ensure that weight is at least eps.
    '''    
    for u, v, d in G.edges(data=True):
        L = float(d.get(length_attr, 1.0))
        if not math.isfinite(L):
            L = 1.0
        w = math.exp(-beta * max(L, 0.0))
        d[weight_attr] = max(w, eps)

def louvain_partition(
    G: nx.Graph,
    weight_attr: str = "weight",
    resolution: float = 1.0,
    seed: int = 0,
    ) -> dict:
    '''
    Perform Louvain community detection on the graph.
    - Inputs:
        - G: nx.Graph
        - weight_attr: str
        - resolution: float
        - seed: int
    - Outputs:
        - part: dict (node to community ID mapping)
    - Procedure:
        Use NetworkX's louvain_communities function to detect communities.
    '''
    comms = nx.algorithms.community.louvain_communities(
        G, weight=weight_attr, resolution=resolution, seed=seed
    )
    part = {}
    for cid, nodes in enumerate(comms):
        for n in nodes:
            part[n] = cid
    return part

def partition_by_igraph(G: nx.Graph, 
                        method: str,
                        seed: int = None) -> list:
    '''
    Partition the graph using igraph community detection methods.
    - Inputs:
        - G: nx.Graph
        - method: str (one of "spinglass", "infomap", "fastgreedy", "edgebetweenness", "label_propagation")
        - seed: int
    - Outputs:
        - labels: list (community labels for each node)
    - Procedure:
        Convert NetworkX graph to igraph.
        Apply the specified community detection method.
    '''
    if seed is not None:
        ig.set_random_number_generator(random.Random(seed))
        print(f"[igraph] set igraph random seed: {seed}")

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)

    if method == "spinglass":
        if not g.is_connected():
            comp = g.components().giant()
            clusters = comp.community_spinglass(spins=10)
            orig_vids = comp.vs.indices

            membership_full = [-1] * g.vcount()
            for local_vid, cid in enumerate(clusters.membership):
                membership_full[orig_vids[local_vid]] = cid
            return np.array(membership_full)

        else:
            clusters = g.community_spinglass(spins=10)
            return np.array(clusters.membership)
    if method == "infomap":
        clusters = g.community_infomap()
    elif method == "fastgreedy":
        clusters = g.community_fastgreedy().as_clustering()
    elif method == "edgebetweenness":
        clusters = g.community_edge_betweenness().as_clustering()
    elif method == "label_propagation":
        clusters = g.community_label_propagation()
    return np.array(clusters.membership) ## return labels

