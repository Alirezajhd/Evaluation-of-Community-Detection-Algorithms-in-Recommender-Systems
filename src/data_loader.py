"""
data_loader.py
--------------
Loads the official CiaoDVD dataset (Guo, Zhang, Thalmann & Yorke-Smith,
"ETAF: An Extended Trust Antecedents Framework for Trust Prediction",
ASONAM 2014), distributed via LibRec (guoguibing.github.io/librec).

Builds:
  - the user-user trust graph (networkx.Graph, undirected -- we
    symmetrize the trust relation since Louvain / Label Propagation /
    BIGCLAM as used in the paper all operate on undirected social graphs)
  - the user-item rating matrix (as a list of (user, item, rating) triples)

File formats (from the dataset's own readme.txt):
  movie-ratings.txt : userID, movieID, genreID, reviewID, movieRating, date  (comma-separated)
  trusts.txt        : trustorID, trusteeID, trustValue                       (comma-separated)
"""
import numpy as np
import networkx as nx


def load_ratings(path):
    """Returns a (N, 3) float array of (user_id, item_id, rating)."""
    raw = np.loadtxt(path, delimiter=",", usecols=(0, 1, 4))
    return raw  # columns: user_id, item_id, rating


def load_trust_graph(path):
    """Returns an undirected networkx.Graph of the trust relations."""
    raw = np.loadtxt(path, delimiter=",", usecols=(0, 1)).astype(int)
    G = nx.Graph()
    G.add_edges_from(raw.tolist())
    return G


def build_reindexed_dataset(rating_path, trust_path):
    """
    Reindexes user/item ids to contiguous 0..N-1 / 0..M-1 ranges (needed for
    the SVD-style latent factor matrices), and restricts the trust graph
    to users that also appear in the rating data (so every community we
    detect can actually be used for the recommendation task).
    """
    ratings = load_ratings(rating_path)
    G_full = load_trust_graph(trust_path)

    rating_users = set(ratings[:, 0].astype(int).tolist())
    trust_users = set(G_full.nodes())
    common_users = rating_users & trust_users

    # keep only ratings from users who appear in the trust graph
    mask = np.isin(ratings[:, 0].astype(int), list(common_users))
    ratings = ratings[mask]

    user_ids = sorted(common_users)
    item_ids = sorted(set(ratings[:, 1].astype(int).tolist()))
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}

    reindexed = np.array(
        [[user2idx[int(u)], item2idx[int(it)], r] for u, it, r in ratings]
    )

    G = nx.Graph()
    G.add_nodes_from(range(len(user_ids)))
    for u, v in G_full.edges():
        if u in common_users and v in common_users:
            G.add_edge(user2idx[u], user2idx[v])

    return {
        "ratings": reindexed,          # (N,3) array: user_idx, item_idx, rating
        "graph": G,                    # trust graph on user_idx nodes
        "n_users": len(user_ids),
        "n_items": len(item_ids),
        "user2idx": user2idx,
        "item2idx": item2idx,
    }


if __name__ == "__main__":
    data = build_reindexed_dataset(
        "data/ciaodvd_movie_ratings.txt", "data/ciaodvd_trusts.txt"
    )
    print("Users:", data["n_users"])
    print("Items:", data["n_items"])
    print("Ratings:", data["ratings"].shape[0])
    print("Trust edges:", data["graph"].number_of_edges())
    print("Isolated users (no trust edge):",
          sum(1 for n in data["graph"].nodes() if data["graph"].degree(n) == 0))

