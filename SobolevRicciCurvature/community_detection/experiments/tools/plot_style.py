import matplotlib.pyplot as plt

METHOD_ORDER = [
    "orc",
    "src-mst",
    "src-spt",
    "infomap",
    "spinglass",
    "fast_greedy",
    "edgebetweenness",
    "label_propagation",
]
tab10 = plt.get_cmap("tab10").colors
COLOR = {
    "orc": tab10[0],              # blue
    "src-mst": tab10[1],          # orange
    "src-spt": tab10[2],          # green

    "edgebetweenness": tab10[3],  # red
    "infomap": tab10[4],          # purple
    "label_propagation": tab10[5],# brown
    "spinglass": tab10[6],        # pink

    "fast_greedy": tab10[7],      # gray (if you use it)
    "fastgreedy": tab10[7],
}

MARKER = {
    "orc": "o",
    "src-mst": "s",
    "src-spt": "D",
    "infomap": "v",
    "spinglass": "X",
    "fast_greedy": "^",
    "fastgreedy": "^",
    "edgebetweenness": "P",
    "label_propagation": "*",
}

LINESTYLE = {
    "orc": "-",
    "src-mst": "--",
    "src-spt": "-.",
    "infomap": "-",
    "spinglass": "--",
    "fast_greedy": ":",
    "fastgreedy": ":",
    "edgebetweenness": ":",
    "label_propagation": "-.",
}

def normalize_method(m: str) -> str:
    return {"fastgreedy": "fast_greedy"}.get(m, m)

def method_rank(m: str) -> int:
    m = normalize_method(m)
    return METHOD_ORDER.index(m) if m in METHOD_ORDER else 999

def style_for(method: str) -> dict:
    m = normalize_method(method)
    return dict(
        color=COLOR.get(m, "black"),
        marker=MARKER.get(m, "o"),
        linestyle=LINESTYLE.get(m, "-"),
    )
