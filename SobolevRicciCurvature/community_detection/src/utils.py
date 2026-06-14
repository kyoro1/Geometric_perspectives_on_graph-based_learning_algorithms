from typing import Optional
import networkx as nx

# =========================
# Graph IO helpers
# =========================
def ensure_edge_lengths(
    G: nx.Graph,
    default_len: float = 1.0,
    attr: str = "length",
    length_attr: Optional[str] = None,
) -> None:
    """Backward/forward compatible initializer."""
    key = length_attr if length_attr is not None else attr
    for u, v, data in G.edges(data=True):
        if key not in data:
            data[key] = float(data.get("weight", default_len))