# Network Analysis Project (v3)
# Title: Influencer Detection, Link Prediction, and Community Discovery in Directed Graphs
'''
This script contains three analytical blocks:
1. **Influencer Detection** – random‑walk with teleportation to find the most visited node.
2. **Link Prediction** – linear‑combination (least‑squares) reconstruction of adjacency rows.
3. **Community Detection** – Louvain modularity on an undirected projection of the graph.

Run:
    pip install networkx matplotlib numpy python-louvain'''


import csv
import random
from collections import defaultdict

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Community package (python‑louvain)
try:
    import community as community_louvain  # pip install python-louvain
except ImportError as e:
    raise ImportError(
        "python-louvain is required. Install via `pip install python-louvain`."\
        f"\nOriginal error: {e}"
    )

###############################################################################
# 1. LOAD GRAPH
###############################################################################

def load_directed_graph(filename: str) -> nx.DiGraph:
    """Read a CSV file and return a directed NetworkX graph."""
    G = nx.DiGraph()
    with open(filename, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row if present
        for row in reader:
            src = row[0].strip()
            for tgt in row[1:]:
                tgt = tgt.strip()
                if tgt:
                    G.add_edge(src, tgt)
    return G

###############################################################################
# 2. INFLUENCER VIA RANDOM WALK WITH TELEPORTATION
###############################################################################

def top_leader_random_walk(
    G: nx.DiGraph, *, steps: int = 100_000, teleport_prob: float = 0.15
) -> str:
    """Return the node visited most often in a random walk with teleportation."""
    nodes = list(G.nodes)
    visits = defaultdict(int)
    current = random.choice(nodes)
    for _ in range(steps):
        if random.random() < teleport_prob:
            current = random.choice(nodes)
        else:
            nbrs = list(G.successors(current))
            current = random.choice(nbrs) if nbrs else random.choice(nodes)
        visits[current] += 1
    return max(visits, key=visits.get)

###############################################################################
# 3. LINK PREDICTION VIA LINEAR COMBINATIONS (LEAST SQUARES)
###############################################################################

def predict_links_linear_combinations(
    adj: np.ndarray, *, threshold: float = 0.14
) -> np.ndarray:
    """Boolean matrix where True indicates a predicted new edge."""
    n = adj.shape[0]
    preds = np.zeros((n, n), dtype=bool)
    for i in range(n):
        target = adj[i]
        others = np.vstack([adj[j] for j in range(n) if j != i])
        x, *_ = np.linalg.lstsq(others.T, target, rcond=None)
        recon = others.T @ x
        preds[i] = (np.abs(recon) > threshold) & (adj[i] == 0)
    np.fill_diagonal(preds, False)
    return preds

###############################################################################
# 4. COMMUNITY DETECTION (LOUVAIN)
###############################################################################

def louvain_communities(G: nx.DiGraph) -> dict[int, list[str]]:
    """Detect communities with Louvain and return cid -> members list."""
    undirected = G.to_undirected()
    partition = community_louvain.best_partition(undirected)  # node -> cid
    comms: dict[int, list[str]] = defaultdict(list)
    for node, cid in partition.items():
        comms[cid].append(node)
    return comms

###############################################################################
# 5. UTILS – SUMMARY & VISUALISATION
###############################################################################

def graph_summary(G: nx.DiGraph):
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Density:", nx.density(G))
    print("Is strongly connected?", nx.is_strongly_connected(G))


def plot_graph(G: nx.DiGraph, comms: dict[int, list[str]] | None = None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    if comms is None:
        nx.draw(G, pos, node_size=20, arrows=False, alpha=0.6)
    else:
        cmap = plt.cm.get_cmap("tab20", len(comms))
        for cid, members in comms.items():
            nx.draw_networkx_nodes(
                G, pos, nodelist=members, node_size=20, node_color=[cmap(cid)]
            )
        nx.draw_networkx_edges(G, pos, arrows=False, alpha=0.3)
    plt.axis("off")
    plt.title("Network (colored by community)" if comms else "Network Graph")
    plt.tight_layout()
    plt.show()

###############################################################################
# 6. MAIN EXECUTION
###############################################################################

if __name__ == "__main__":
    CSV_PATH = "data"  # <-- Updated file name

    # 6.1 Load graph and summary
    G = load_directed_graph(CSV_PATH)
    graph_summary(G)

    # 6.2 Influencer Detection
    print("\n--- Part 1: Influencer Detection (Random Walk) ---")
    leader = top_leader_random_walk(G)
    print("Top leader node:", leader)

    # 6.3 Link Prediction
    print("\n--- Part 2: Link Prediction (Least Squares) ---")
    adj = nx.to_numpy_array(G, dtype=float)
    preds = predict_links_linear_combinations(adj)
    node_list = list(G.nodes)
    print("Number of predicted links:", int(preds.sum()))
    for i, src in enumerate(node_list):
        for j, dst in enumerate(node_list):
            if preds[i, j]:
                print(f"{src} -> {dst}")

    # 6.4 Community Detection
    print("\n--- Part 3: Community Detection (Louvain) ---")
    communities = louvain_communities(G)
    for cid, members in communities.items():
        print(f"Community {cid} (size={len(members)}): {members}")

    # 6.5 Optional visualisation (comment out if running headless)
    try:
        plot_graph(G, communities)
    except Exception as e:
        print("Visualization skipped:", e)

