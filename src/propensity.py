"""
propensity.py
-------------
Computes the propensity alpha_uc of user u within community c, as
described in Sec. 4 of the paper: alpha_uc is a centrality measure of u
computed *inside the induced subgraph of community c* (not the whole
trust graph). We support the three centrality types the paper compares:
degree, betweenness, closeness.

alpha_uc is used directly as the raw (0..1) centrality value returned by
networkx's centrality functions -- this matches the paper's description
literally ("alpha_uc a centrality of u within the community subgraph c").

Note: an earlier version of this module additionally renormalized each
user's propensities to sum to 1 across their communities. That is WRONG
for non-overlapping methods (Louvain, Label Propagation): since every
user belongs to exactly one community there, sum-to-1 normalization
forces alpha_uc = 1.0 for every single user, completely erasing the
centrality signal the paper is trying to capture (and making Louvain and
Label Propagation results identical across degree/betweenness/closeness,
which they must NOT be). We keep alpha_uc as the raw centrality value.
For BIGCLAM (overlapping), a user's total community-influence term in
ConSVD is naturally the sum of alpha_uc * p_c over the (possibly several)
communities they belong to -- no extra normalization is needed for that
to be well-defined.
"""
import networkx as nx


CENTRALITY_FUNCS = {
    "degree": nx.degree_centrality,
    "betweenness": nx.betweenness_centrality,
    "closeness": nx.closeness_centrality,
}


def compute_propensities(G, communities, centrality_type):
    """
    Returns dict[(user, community_id)] -> alpha_uc, the raw centrality of
    user u inside the induced subgraph of community c.
    """
    if centrality_type not in CENTRALITY_FUNCS:
        raise ValueError(f"Unknown centrality type: {centrality_type}")
    cfun = CENTRALITY_FUNCS[centrality_type]

    alpha = {}
    for cid, members in communities.items():
        if len(members) == 1:
            (only_user,) = tuple(members)
            alpha[(only_user, cid)] = 1.0
            continue
        sub = G.subgraph(members)
        cvals = cfun(sub)
        for u, val in cvals.items():
            alpha[(u, cid)] = val
    return alpha


if __name__ == "__main__":
    from data_loader import build_reindexed_dataset
    from community_detection import run_louvain, run_bigclam

    data = build_reindexed_dataset(
        "data/ciaodvd_movie_ratings.txt", "data/ciaodvd_trusts.txt"
    )
    G = data["graph"]

    comms, memb = run_louvain(G)
    biggest_cid = max(comms, key=lambda c: len(comms[c]))
    sample_users = list(comms[biggest_cid])[:5]
    print(f"Louvain, biggest community size={len(comms[biggest_cid])}")
    for ctype in ["degree", "betweenness", "closeness"]:
        alpha = compute_propensities(G, comms, ctype)
        vals = [round(alpha[(u, biggest_cid)], 5) for u in sample_users]
        print(f"  {ctype}: {vals}")

    comms_b, memb_b = run_bigclam(G)
    sample_user = next(u for u, cs in memb_b.items() if len(cs) > 1)
    alpha_b = compute_propensities(G, comms_b, "closeness")
    print(f"BIGCLAM overlap check: user {sample_user} in communities "
          f"{memb_b[sample_user]}, alpha values: "
          f"{[round(alpha_b[(sample_user, c)], 4) for c in memb_b[sample_user]]}")
