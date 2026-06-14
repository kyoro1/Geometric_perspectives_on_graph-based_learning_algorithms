import os
import networkx as nx
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import time
import urllib.request
import urllib.error
import gzip

import shutil
import zipfile
from dataclasses import dataclass

@dataclass
class DatasetSpec:
    name: str
    kind: str  # "builtin", "zip", "gml", "csv"
    url: Optional[str] = None

DATASETS: Dict[str, DatasetSpec] = {
    "karate": DatasetSpec("karate", "builtin"),
    "football": DatasetSpec("football", "zip", "https://websites.umich.edu/~mejn/netdata/football.zip"),
    "polbooks": DatasetSpec("polbooks", "zip", "https://websites.umich.edu/~mejn/netdata/polbooks.zip"),
    "polblogs": DatasetSpec("polblogs", "zip", "https://websites.umich.edu/~mejn/netdata/polblogs.zip"),
    "email-eu-core": DatasetSpec("email-eu-core","snap_gz",url="https://snap.stanford.edu/data/email-Eu-core.txt.gz"),
    }

class DataLoader():
    def __init__(self,
                n: int = 500,
                seed: int = 0,
                 ):
        self.n = n
        self.seed = seed # For random seed
        self.name = None # For real datasets

    ## SBM SYNTHETIC DATASET ##
    def generate_sbm(self,
                    pintra: float = 0.15,
                    ratio: float = 0.5,  # pinter/pintra
                     ) -> Tuple[nx.Graph, Dict[int, int]]:
        """
        Simple SBM with 2 equal-sized communities.
        - Inputs:
            - pintra: float, intra-community edge probability
            - ratio: float, inter-community edge probability ratio (pinter/pintra)
        - Outputs:
            - G: nx.Graph, generated graph
        - Process:
            - Uses numpy random generator with self.seed for reproducibility.
            - Creates 2 communities of sizes n//2 and n - (n//2).
            - Assigns ground truth labels 0 and 1 to the two communities.
            - Returns the generated graph and ground truth labels.
        """
        rng = np.random.default_rng(self.seed)
        sizes = [self.n // 2, self.n - (self.n // 2)]
        pinter = float(ratio) * float(pintra)
        p = [
            [pintra, pinter],
            [pinter, pintra],
        ]
        G = nx.stochastic_block_model(sizes, p, seed=int(rng.integers(0, 2**31 - 1)))
        G = self.to_simple_undirected(G)
        # ground truth labels
        gt: Dict[int, int] = {}
        # nodes are 0..n-1 in order
        for i in range(sizes[0]):
            gt[i] = 0
        for i in range(sizes[0], self.n):
            gt[i] = 1
        return G, gt

    @staticmethod
    def to_simple_undirected(G: nx.Graph) -> nx.Graph:
        """Convert to simple undirected graph (remove parallel edges, self-loops)."""
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            H = nx.Graph()
            H.add_nodes_from(G.nodes(data=True))
            for u, v, d in G.edges(data=True):
                if u == v:
                    continue
                if H.has_edge(u, v):
                    continue
                H.add_edge(u, v, **d)
            G = H
        if G.is_directed():
            G = nx.Graph(G)
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        return G

    ## LFR SYNTHETIC DATASET ##
    def generate_lfr(self,
        mu: float = 0.1,
        avg_deg: int = 20,
        seed_tries: int = 200,
        tau1: float = 3.0,
        tau2: float = 1.5,
        min_community: int = 20,
        max_community: int = 100,
        max_degree: int = 50,
        keep_giant_component: bool = True,
        overlap_policy: str = "largest",  # "largest" | "smallest" | "error"
    ) -> Tuple[nx.Graph, Dict[int, int]]:
        """
        Generate LFR benchmark graph with ground truth communities.
        - Inputs:
            - mu: float, mixing parameter (0.0 = strong communities, 1.0 = random)
            - avg_deg: int, average degree
            - seed_tries: int, number of seed attempts for generation
            - tau1: float, power-law exponent for degree distribution
            - tau2: float, power-law exponent for community size distribution
            - min_community: int, minimum community size
            - max_community: int, maximum community size
            - max_degree: int, maximum degree
            - keep_giant_component: bool, whether to keep only the giant connected component
            - overlap_policy: str, how to handle overlapping communities ("largest", "smallest", "error")
        - Outputs:
            - G: nx.Graph, generated graph
            - gt: Dict[int, int], ground truth community labels (node -> community id)
        - Process:
            - Tries to generate the LFR graph up to seed_tries times with different seeds.
            - Extracts and resolves community assignments based on overlap_policy.
            - Optionally keeps only the giant connected component.
            - Returns the generated graph and ground truth labels.
        """
        mu = float(mu)
        last_err: Optional[Exception] = None

        for s_off in range(seed_tries):
            s = int(self.seed) + s_off
            try:
                # --- Generate raw LFR ---
                G = nx.LFR_benchmark_graph(
                    self.n,
                    tau1=tau1,
                    tau2=tau2,
                    mu=mu,
                    average_degree=avg_deg,
                    min_community=min_community,
                    max_community=max_community,
                    max_degree=max_degree,
                    seed=s,
                )

                G = nx.Graph(G)
                G.remove_edges_from(nx.selfloop_edges(G))

                # --- Extract communities attribute ---
                comm_attr = nx.get_node_attributes(G, "community")
                if len(comm_attr) == 0:
                    raise RuntimeError("No 'community' attribute found in generated LFR graph.")

                def resolve_comm(c):
                    if c is None:
                        return None

                    # Case A: single community represented as set/frozenset of nodes
                    if isinstance(c, (set, frozenset)) and all(isinstance(x, int) for x in c):
                        return frozenset(c)

                    # Case B: overlap represented as set of communities (each itself set/frozenset)
                    if isinstance(c, set) and all(isinstance(x, (set, frozenset)) for x in c):
                        comms = [frozenset(x) for x in c]

                        if overlap_policy == "error":
                            raise RuntimeError("Overlapping communities detected.")

                        if overlap_policy == "largest":
                            # choose largest; tie-breaker: smallest min node id
                            comms.sort(key=lambda z: (-len(z), min(z)))
                            return comms[0]

                        if overlap_policy == "smallest":
                            # choose smallest; tie-breaker: smallest min node id
                            comms.sort(key=lambda z: (len(z), min(z)))
                            return comms[0]

                        raise ValueError(f"Unknown overlap_policy={overlap_policy}")

                    # Fallback: try to coerce
                    try:
                        return frozenset(c)
                    except Exception:
                        raise RuntimeError(f"Unexpected community attribute type: {type(c)} -> {c}")

                node2commset = {}
                for u, c in comm_attr.items():
                    cc = resolve_comm(c)
                    if cc is not None:
                        node2commset[u] = cc
                if keep_giant_component and not nx.is_connected(G):
                    giant = max(nx.connected_components(G), key=len)
                    G = G.subgraph(giant).copy()
                    node2commset = {u: node2commset[u] for u in G.nodes() if u in node2commset}

                uniq_commsets = sorted(
                    set(node2commset.values()),
                    key=lambda z: (len(z), min(z))
                )
                comm2id = {cset: i for i, cset in enumerate(uniq_commsets)}

                gt = {u: comm2id[node2commset[u]] for u in G.nodes() if u in node2commset}

                # --- Optional sanity checks ---
                # Ensure every node has a label (it should, but safety)
                if len(gt) != G.number_of_nodes():
                    # You can choose to raise instead.
                    # For reproduction, it's often safer to raise to avoid silent mismatch.
                    missing = set(G.nodes()) - set(gt.keys())
                    raise RuntimeError(f"Some nodes missing gt labels: {len(missing)} nodes")

                return G, gt

            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(
            f"LFR generation failed after seed_tries={seed_tries} (base_seed={self.seed}, mu={mu}). "
            f"Last error: {last_err}"
        )

    def load_real_graph(
        self,
        name: str,
        data_dir: str = "./data",
    ) -> Tuple[nx.Graph, Optional[Dict[int, Any]]]:
        """
        Load real-world graph datasets with ground truth communities.
        - Inputs:
            - name: str, dataset name ("karate", "football", "polbooks", "polblogs", "email-eu-core")
            - data_dir: str, base directory for dataset storage
        - Outputs:
            - G: nx.Graph, loaded graph
            - labels: Optional[Dict[int, Any]], ground truth labels (node -> label) if available
        - Process:
            - Supports built-in datasets and downloads from specified URLs if not cached.
            - Handles dataset-specific loading and preprocessing.
       """
        self.name = name.lower()
        print(f"Loading real graph: {self.name}")
        if self.name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}. available={list(DATASETS.keys())}")

        if self.name == "karate":
            G = nx.karate_club_graph()
            G = self.to_simple_undirected(G)
            labels = {n: (0 if G.nodes[n].get("club") == "Mr. Hi" else 1) for n in G.nodes()}
            return nx.Graph(G), labels

        if self.name in ("football", "polbooks", "polblogs"):
            return self._fetch_zip_mejn_netdata(data_dir=data_dir)
        if self.name == "email-eu-core":
            return self._fetch_email_eu_core(data_dir=data_dir)
        raise NotImplementedError(self.name)

    def _fetch_zip_mejn_netdata(self, data_dir: str) -> Tuple[nx.Graph, Optional[Dict[int, Any]]]:
        """
        Handles football, polbooks, polblogs.
        - Tries local cache first (gml / extracted zip)
        - If not found, downloads from candidates
        - Falls back to GitHub raw .gml mirrors if needed
        """
        import zipfile
        import shutil

        spec = DATASETS[self.name]
        assert spec.url is not None

        base = os.path.join(data_dir, self.name)
        self._mkdir(base)

        zip_path = os.path.join(base, f"{self.name}.zip")
        extract_dir = os.path.join(base, "raw")

        # ----------------------------------------------------
        # 0) Try local cached files first
        # ----------------------------------------------------
        # Case A: direct gml exists
        gml_path: Optional[str] = os.path.join(base, f"{self.name}.gml")
        if os.path.exists(gml_path):
            print(f"Found cached GML: {gml_path}")
        else:
            gml_path = None

            # Case B: extracted dir exists and contains .gml
            if os.path.exists(extract_dir):
                for root, _, files in os.walk(extract_dir):
                    for fn in files:
                        if fn.lower().endswith(".gml"):
                            gml_path = os.path.join(root, fn)
                            print(f"Found extracted GML: {gml_path}")
                            break
                    if gml_path:
                        break

            # Case C: zip exists but not extracted (or extracted missing)
            if (gml_path is None) and os.path.exists(zip_path) and zipfile.is_zipfile(zip_path):
                print(f"Found cached ZIP: {zip_path} -> extracting")
                if os.path.exists(extract_dir):
                    # keep it simple: clear stale extraction to avoid partial files
                    shutil.rmtree(extract_dir)
                self._extract_zip(zip_path, extract_dir)

                for root, _, files in os.walk(extract_dir):
                    for fn in files:
                        if fn.lower().endswith(".gml"):
                            gml_path = os.path.join(root, fn)
                            print(f"Found GML in cached ZIP: {gml_path}")
                            break
                    if gml_path:
                        break

        if gml_path is None:
            # ----------------------------------------------------
            # 1) Download from candidates if cache not available
            # ----------------------------------------------------
            candidates: List[str] = [spec.url]

            if self.name in ("football", "polbooks", "polblogs"):
                umich_https = f"https://websites.umich.edu/~mejn/netdata/{self.name}.zip"
                if umich_https not in candidates:
                    candidates.append(umich_https)

            if self.name == "football":
                candidates += [
                    "https://raw.githubusercontent.com/Social-Network-Analysis/Link-Prediction-Algorithms/master/datasets/football.gml",
                ]
            elif self.name == "polbooks":
                candidates += [
                    "https://raw.githubusercontent.com/patapizza/pylouvain/master/data/polbooks.gml",
                ]
            elif self.name == "polblogs":
                candidates += [
                    "https://raw.githubusercontent.com/AMarz4400/Political_Blogs_Classification_GNN/main/polblogs.gml",
                ]

            last_err: Optional[Exception] = None

            for url in candidates:
                try:
                    # ---- Direct GML mirror ----
                    if url.lower().endswith(".gml"):
                        gml_path = os.path.join(base, f"{self.name}.gml")
                        if not os.path.exists(gml_path):
                            self.download(url, gml_path, overwrite=False)
                        print(f"Fetched GML directly: {url}")
                        break

                    # ---- ZIP case ----
                    if not os.path.exists(zip_path):
                        self.download(url, zip_path, overwrite=False)
                    print(f"Fetched ZIP: {url}")
                    print("zip size:", os.path.getsize(zip_path), "is_zipfile:", zipfile.is_zipfile(zip_path))

                    if not zipfile.is_zipfile(zip_path):
                        raise RuntimeError("Downloaded file is not a ZIP (maybe HTML/blocked).")

                    # clean extract dir each time to avoid stale partial files
                    if os.path.exists(extract_dir):
                        shutil.rmtree(extract_dir)
                    self._extract_zip(zip_path, extract_dir)

                    # find .gml inside
                    for root, _, files in os.walk(extract_dir):
                        for fn in files:
                            if fn.lower().endswith(".gml"):
                                gml_path = os.path.join(root, fn)
                                break
                        if gml_path:
                            break

                    if gml_path:
                        print("Found GML in ZIP:", gml_path)
                        break
                    else:
                        raise RuntimeError("ZIP extracted but no .gml found inside.")

                except Exception as e:
                    last_err = e
                    print("FAILED:", url)
                    print("  error:", repr(e))
                    continue

            if gml_path is None:
                raise RuntimeError(f"Could not fetch {self.name} gml from any candidate. last_err={last_err!r}")

        # ----------------------------------------------------
        # 2) Read graph & labels (same as original)
        # ----------------------------------------------------
        G = self._read_gml_as_undirected_int_nodes(gml_path)
        G = self.lcc(G)

        labels: Dict[int, Any] = {}
        for n, d in G.nodes(data=True):
            if "value" in d:
                labels[n] = d["value"]
            elif "group" in d:
                labels[n] = d["group"]
            elif "party" in d:
                labels[n] = d["party"]
            elif "congress" in d:
                labels[n] = d["congress"]

        if len(labels) == 0:
            labels = None

        if labels is not None:
            try:
                _ = [int(v) for v in labels.values()]
                labels = {k: int(v) for k, v in labels.items()}
            except Exception:
                uniq = sorted(set(labels.values()), key=lambda x: str(x))
                mapping = {v: i for i, v in enumerate(uniq)}
                labels = {k: mapping[v] for k, v in labels.items()}
                print("Label mapping:", mapping)

        return G, labels

    def _mkdir(self,
               p: str) -> None:
        if p and (not os.path.exists(p)):
            os.makedirs(p, exist_ok=True)

    def download(self,
                url: str,
                dst_path: str,
                overwrite: bool = False,
                user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                retries: int = 3,
                backoff_sec: float = 1.0,
                ) -> str:
        self._mkdir(os.path.dirname(dst_path))
        if os.path.exists(dst_path) and (not overwrite):
            return dst_path

        tmp = dst_path + ".tmp"
        last_err: Optional[Exception] = None

        for i in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": user_agent})
                with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                os.replace(tmp, dst_path)
                return dst_path
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                last_err = e
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                time.sleep(backoff_sec * (2 ** i))

        raise RuntimeError(f"download failed: {url} -> {dst_path}, last_err={last_err!r}")

    def _extract_zip(self,
                     zip_path: str, 
                     out_dir: str) -> None:
        self._mkdir(out_dir)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)

    def _fetch_email_eu_core(self, data_dir: str) -> Tuple[nx.Graph, Optional[Dict[int, Any]]]:
        """
        SNAP email-Eu-core
        - edges: email-Eu-core.txt.gz
        - labels: email-Eu-core-department-labels.txt.gz

        Behavior:
        - Try local cached files first
        - Download only missing files (no overwrite)
        """
        base = os.path.join(data_dir, "email-eu-core")
        self._mkdir(base)

        edge_url  = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"
        label_url = "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"

        edge_gz  = os.path.join(base, "email-Eu-core.txt.gz")
        label_gz = os.path.join(base, "email-Eu-core-department-labels.txt.gz")

        # ----------------------------------------------------
        # 0) Try cache first; download only if missing
        # ----------------------------------------------------
        if not os.path.exists(edge_gz):
            self.download(edge_url, edge_gz, overwrite=False)
        else:
            print(f"Found cached edge file: {edge_gz}")

        if not os.path.exists(label_gz):
            self.download(label_url, label_gz, overwrite=False)
        else:
            print(f"Found cached label file: {label_gz}")

        # ----------------------------------------------------
        # 1) Read edges (gz) (same as original)
        # ----------------------------------------------------
        edges = []
        with gzip.open(edge_gz, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                a, b = line.split()
                edges.append((int(a), int(b)))

        G = nx.Graph()
        G.add_edges_from(edges)
        G = self.to_simple_undirected(G)

        print("RAW graph:", G.number_of_nodes(), G.number_of_edges())
        print("RAW isolates:", len(list(nx.isolates(G))))
        print("RAW components:", nx.number_connected_components(G.to_undirected()))

        # Keep largest connected component and relabel nodes to 0..N-1
        G = self.lcc(G)
        print("LCC graph:", G.number_of_nodes(), G.number_of_edges())
        G = nx.convert_node_labels_to_integers(G, label_attribute="orig")

        # ----------------------------------------------------
        # 2) Read labels (gz) (same as original)
        # ----------------------------------------------------
        labels_raw: Dict[int, int] = {}
        with gzip.open(label_gz, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                a, b = line.split()
                labels_raw[int(a)] = int(b)

        # map labels to relabeled nodes
        labels: Dict[int, int] = {}
        for new_id, d in G.nodes(data=True):
            orig = int(d["orig"])
            if orig in labels_raw:
                labels[new_id] = labels_raw[orig]

        if len(labels) == 0:
            labels = None

        return G, labels

    def lcc(self, G: nx.Graph) -> nx.Graph:
        """Largest connected component."""
        if G.number_of_nodes() == 0:
            return G
        comps = list(nx.connected_components(G))
        comps.sort(key=len, reverse=True)
        keep = comps[0]
        return G.subgraph(keep).copy()

    def _read_gml_as_undirected_int_nodes(self, 
                                          gml_path: str) -> nx.Graph:
        G = nx.read_gml(gml_path)
        G = self.to_simple_undirected(G)
        return nx.convert_node_labels_to_integers(G, label_attribute="orig")

    def calc_max_lcc_size(self, G: nx.Graph) -> int:
        H = G.to_undirected() if G.is_directed() else G
        return max((len(c) for c in nx.connected_components(H)), default=0)

    def summarize_graph(self, G: nx.Graph, name: str = ""):
        H = G.to_undirected() if G.is_directed() else G

        n = G.number_of_nodes()
        m = G.number_of_edges()
        directed = G.is_directed()

        ncc = nx.number_connected_components(H)
        lcc_size = max((len(c) for c in nx.connected_components(H)), default=0)

        density = nx.density(H)
        avg_deg = (2*m / n) if n > 0 else 0.0

        print("="*60)
        if name:
            print(f"[{name}]")
        print(f"nodes          : {n}")
        print(f"edges          : {m}")
        print(f"directed       : {directed}")
        print(f"#components    : {ncc}")
        print(f"LCC size       : {lcc_size}")
        print(f"avg degree     : {avg_deg:.3f}")
        print(f"density        : {density:.6f}")
        print("="*60)