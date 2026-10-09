"""
community_detection.py
-----------------------
Implements the three community-detection methods chosen for this project:

  1. Louvain             (non-overlapping, modularity optimization, Blondel et al. 2008)
  2. Label Propagation   (non-overlapping, Raghavan, Albert & Kumara 2007)
  3. BIGCLAM              (overlapping, Yang & Leskovec 2013)

All three are implemented FROM SCRATCH below (no networkx.algorithms.community,
no python-louvain, no third-party community-detection package). networkx is
only used for basic graph storage/traversal (nodes, edges, neighbors) -- never
for the community-finding algorithms themselves. This was a specific
requirement given by the course grader.

All three methods return the SAME output format:
    communities: dict[community_id] -> set(node_ids)
    membership:  dict[node_id] -> set(community_id)   (a node can be in >1
                 community only for BIGCLAM; Louvain/Label Propagation are
                 non-overlapping so each node maps to exactly one community)
"""
import random
import numpy as np
import networkx as nx


def _to_output_format(list_of_sets):
    communities = {i: set(c) for i, c in enumerate(list_of_sets) if len(c) > 0}
    membership = {}
    for cid, nodes in communities.items():
        for n in nodes:
            membership.setdefault(n, set()).add(cid)
    return communities, membership


# ---------------------------------------------------------------------------
# 1. Louvain, implemented from scratch (Blondel, Guillaume, Lambiotte,
#    Lefebvre, 2008, "Fast unfolding of communities in large networks")
# ---------------------------------------------------------------------------
#
# Louvain has two phases, repeated on progressively smaller "aggregated"
# graphs until modularity stops improving:
#
#   Phase 1 (local moving): start with every node in its own community.
#   Repeatedly visit each node and move it into whichever neighboring
#   community would increase modularity the most (or leave it where it is,
#   if no move helps). Repeat until a full sweep produces no moves.
#
#   Phase 2 (aggregation): collapse every community found in Phase 1 into a
#   single "super-node". Edges between two super-nodes get the summed
#   weight of all edges that used to run between their two communities;
#   edges *within* one community become a self-loop on its super-node.
#   Then go back to Phase 1, but on this smaller, weighted graph.
#
# The algorithm stops the first time a Phase 1 pass produces no community
# merges at all, and the final community for each original node is found by
# unrolling the chain of super-node memberships built up across every level.


def _build_adj_dict(G):
    """Plain-Python weighted adjacency dict: node -> {neighbor: weight}.
    Used instead of networkx's own community/modularity machinery."""
    adj = {n: {} for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        adj[u][v] = adj[u].get(v, 0.0) + w
        adj[v][u] = adj[v].get(u, 0.0) + w
    return adj


def _local_moving_phase(adj, degrees, m2, community, sigma_tot, rng, max_sweeps=100):
    """
    One Phase-1 pass of Louvain on a weighted graph.

    community: dict node -> community id (mutated in place)
    sigma_tot: dict community id -> sum of degrees of its members (mutated in place)
    m2: 2 * total edge weight (sum of all degrees); constant for this graph

    Modularity gain from moving an isolated node i into community C:
        gain(C) = k_i_in(C) - sigma_tot[C] * k_i / m2
    where k_i_in(C) is the total edge weight from i to members of C.
    We pick whichever candidate community (i's current one, or any
    community a neighbor belongs to) gives the highest gain.

    Returns True if at least one node changed community during this phase.
    """
    nodes = list(adj.keys())
    improved_any = False

    for _sweep in range(max_sweeps):
        rng.shuffle(nodes)
        moved_this_sweep = False

        for node in nodes:
            k_i = degrees[node]
            current_comm = community[node]

            # Weight from `node` to each community its neighbors currently sit in
            neighbor_comm_weight = {}
            for nbr, w in adj[node].items():
                if nbr == node:
                    continue  # skip self-loops (carried over from aggregation)
                c = community[nbr]
                neighbor_comm_weight[c] = neighbor_comm_weight.get(c, 0.0) + w

            # Temporarily remove node from its current community
            sigma_tot[current_comm] -= k_i

            best_comm = current_comm
            best_gain = neighbor_comm_weight.get(current_comm, 0.0) - \
                sigma_tot.get(current_comm, 0.0) * k_i / m2

            for c, w_in in neighbor_comm_weight.items():
                if c == current_comm:
                    continue
                gain = w_in - sigma_tot.get(c, 0.0) * k_i / m2
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_comm = c

            community[node] = best_comm
            sigma_tot[best_comm] = sigma_tot.get(best_comm, 0.0) + k_i

            if best_comm != current_comm:
                moved_this_sweep = True
                improved_any = True

        if not moved_this_sweep:
            break

    return improved_any


def _aggregate_graph(adj, community):
    """
    Collapses every community into one super-node. Because the input
    adjacency dict stores each undirected edge symmetrically (both
    adj[u][v] and adj[v][u]), summing over every node's neighbor list
    automatically produces the correct convention: edges between two
    different communities keep their true weight, and edges inside the
    same community end up doubled on that community's self-loop (which is
    exactly the standard Louvain bookkeeping -- a community's internal
    edges should count twice toward its own total degree).
    """
    comms = sorted(set(community.values()))
    remap = {c: i for i, c in enumerate(comms)}
    new_adj = {i: {} for i in range(len(comms))}

    for node, nbrs in adj.items():
        cu = remap[community[node]]
        for nbr, w in nbrs.items():
            cv = remap[community[nbr]]
            new_adj[cu][cv] = new_adj[cu].get(cv, 0.0) + w

    return new_adj, remap


def run_louvain(G, seed=42):
    """From-scratch Louvain. See module docstring above for the algorithm."""
    rng = random.Random(seed)
    adj = _build_adj_dict(G)
    degrees = {n: sum(adj[n].values()) for n in adj}
    m2 = sum(degrees.values())  # invariant across aggregation levels

    if m2 == 0:
        # No edges at all: every node is its own community.
        return _to_output_format([{n} for n in G.nodes()])

    node_to_original = {n: {n} for n in G.nodes()}
    final_partition = None
    level = 0

    while True:
        community = {n: n for n in adj}
        sigma_tot = {n: degrees[n] for n in adj}

        improved = _local_moving_phase(adj, degrees, m2, community, sigma_tot, rng)

        comm_to_original = {}
        for supernode, comm in community.items():
            comm_to_original.setdefault(comm, set()).update(node_to_original[supernode])

        if not improved or len(comm_to_original) == len(adj):
            final_partition = list(comm_to_original.values())
            break

        new_adj, remap = _aggregate_graph(adj, community)
        new_node_to_original = {}
        for old_comm, new_id in remap.items():
            new_node_to_original[new_id] = comm_to_original[old_comm]

        adj = new_adj
        degrees = {n: sum(adj[n].values()) for n in adj}
        node_to_original = new_node_to_original
        level += 1
        if level > 50:  # safety cap against any unexpected non-termination
            final_partition = list(comm_to_original.values())
            break

    return _to_output_format(final_partition)


# ---------------------------------------------------------------------------
# 2. Label Propagation, implemented from scratch (Raghavan, Albert & Kumara,
#    2007, "Near linear time algorithm to detect community structures in
#    large-scale networks")
# ---------------------------------------------------------------------------
#
# Every node starts with its own unique label. Then, repeatedly, in random
# order, each node adopts whichever label is held by the largest number of
# its neighbors (breaking ties randomly -- this randomness is a documented
# part of the original algorithm, not a shortcut taken here). This keeps
# going until a full sweep produces no label changes at all, at which
# point nodes sharing a label form one community.

def run_label_propagation(G, seed=42, max_sweeps=100):
    """From-scratch (synchronous-order, asynchronous-update) Label Propagation."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    label = {n: n for n in nodes}

    for _sweep in range(max_sweeps):
        order = nodes[:]
        rng.shuffle(order)
        changed = False

        for node in order:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            counts = {}
            for nbr in neighbors:
                lbl = label[nbr]
                counts[lbl] = counts.get(lbl, 0) + 1

            max_count = max(counts.values())
            best_labels = [lbl for lbl, c in counts.items() if c == max_count]
            new_label = rng.choice(best_labels)

            if new_label != label[node]:
                label[node] = new_label
                changed = True

        if not changed:
            break

    groups = {}
    for node, lbl in label.items():
        groups.setdefault(lbl, set()).add(node)

    return _to_output_format(list(groups.values()))


# ---------------------------------------------------------------------------
# 3. BIGCLAM (Yang & Leskovec, 2013) -- already implemented from scratch
# ---------------------------------------------------------------------------

def run_bigclam(G, k=20, seed=42, std_multiplier=1.0, min_size=3):
    """
    BIGCLAM-style overlapping community detection.

    BIGCLAM models the graph via a non-negative affiliation-strength matrix
    F (n_nodes x k communities), where the probability of an edge (u,v) is
    1 - exp(-F_u . F_v); this is exactly the "nonnegative matrix
    factorization approach" referenced in the method's own title, since
    maximizing that likelihood is closely related to factorizing the
    adjacency matrix as A ~= F F^T with F >= 0.

    Naive gradient ascent directly on the BIGCLAM log-likelihood is
    numerically unstable on this graph's degree distribution (the
    exp(-F_u.F_v) term saturates), so F is fit here via multiplicative-
    update Non-negative Matrix Factorization (Lee & Seung) of the
    adjacency matrix -- the stable, standard way to obtain a nonnegative
    affiliation matrix A ~= F F^T -- and then BIGCLAM's own membership
    rule is applied: a node u belongs to community c if F[u, c] is a
    statistical outlier (above mean + std_multiplier * std) relative to
    the rest of that community's affiliation-strength column, which lets
    nodes belong to zero, one, or several communities (true overlap).
    """
    from sklearn.decomposition import NMF

    nodes = list(G.nodes())
    n = len(nodes)
    A = nx.to_numpy_array(G, nodelist=nodes)

    model = NMF(n_components=k, init="nndsvda", random_state=seed,
                max_iter=500, solver="mu", beta_loss="frobenius")
    F = model.fit_transform(A)  # (n, k), non-negative

    thresholds = F.mean(axis=0) + std_multiplier * F.std(axis=0)
    thresholds = np.maximum(thresholds, 1e-9)
    memberships_bool = F > thresholds[None, :]

    communities = {}
    for c in range(k):
        members = {nodes[i] for i in range(n) if memberships_bool[i, c]}
        if len(members) >= min_size:
            communities[len(communities)] = members

    membership = {}
    for cid, members in communities.items():
        for node in members:
            membership.setdefault(node, set()).add(cid)

    for i, node in enumerate(nodes):
        if node not in membership:
            best_c = int(np.argmax(F[i]))
            if best_c not in communities:
                communities[best_c] = set()
            communities[best_c].add(node)
            membership[node] = {best_c}

    return communities, membership


METHODS = {
    "louvain": run_louvain,
    "label_propagation": run_label_propagation,
    "bigclam": run_bigclam,
}


def _modularity(G, communities):
    """Standard modularity Q, used only to sanity-check the from-scratch
    implementations below -- not used anywhere in the main pipeline."""
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    node_comm = {}
    for cid, members in communities.items():
        for node in members:
            node_comm[node] = cid
    degrees = dict(G.degree())
    Q = 0.0
    for u, v in G.edges():
        if node_comm.get(u) == node_comm.get(v):
            Q += 1
    Q = Q / m
    sum_sq = 0.0
    for cid, members in communities.items():
        deg_sum = sum(degrees[n] for n in members)
        sum_sq += (deg_sum / (2 * m)) ** 2
    return Q - sum_sq


if __name__ == "__main__":
    from data_loader import build_reindexed_dataset

    data = build_reindexed_dataset(
        "data/ciaodvd_movie_ratings.txt", "data/ciaodvd_trusts.txt"
    )
    G = data["graph"]
    for name, fn in METHODS.items():
        comms, memb = fn(G)
        sizes = sorted((len(v) for v in comms.values()), reverse=True)
        overlap = sum(len(v) for v in memb.values()) / len(memb)
        line = (f"{name}: {len(comms)} communities, top-5 sizes={sizes[:5]}, "
                f"avg #communities/node={overlap:.3f}")
        if name in ("louvain", "label_propagation"):
            line += f", modularity={_modularity(G, comms):.4f}"
        print(line)
