"""
evaluate.py
-----------
Runs the full experiment grid: for each of the 3 chosen community
detection methods (Louvain, Label Propagation, BIGCLAM) x each of the 3
centrality-based propensities (degree, betweenness, closeness), train
ConSVD with 5-fold cross-validation on the Ciao rating data and report
average RMSE / MAE, replicating the structure of Table 2 in the paper.

Community detection is run ONCE per method on the full trust graph
(communities are a property of the social network, not of the train/test
rating split -- this matches how the paper describes the framework: CD
runs once, then features/propensities feed the recommendation model).
"""
import json
import numpy as np
from data_loader import build_reindexed_dataset
from community_detection import METHODS
from propensity import compute_propensities
from consvd import ConSVD


def k_fold_indices(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return folds


def run_experiment_grid(rating_path, trust_path, cd_methods, centrality_types,
                         k_folds=5, d=20, n_epochs=15):
    data = build_reindexed_dataset(rating_path, trust_path)
    G = data["graph"]
    ratings = data["ratings"]
    n_users, n_items = data["n_users"], data["n_items"]

    print(f"Dataset: {n_users} users, {n_items} items, {len(ratings)} ratings, "
          f"{G.number_of_edges()} trust edges\n")

    # 1. Run each community detection method ONCE on the full trust graph
    cd_results = {}
    for name in cd_methods:
        print(f"Running community detection: {name} ...")
        communities, membership = METHODS[name](G)
        cd_results[name] = (communities, membership)
        print(f"  -> {len(communities)} communities, "
              f"avg communities/user = {np.mean([len(c) for c in membership.values()]):.2f}")

    folds = k_fold_indices(len(ratings), k=k_folds)
    results = []  # list of dicts: method, centrality, fold, rmse, mae

    for cd_name in cd_methods:
        communities, membership = cd_results[cd_name]
        for centrality in centrality_types:
            print(f"\n=== {cd_name} + {centrality} centrality ===")
            alpha = compute_propensities(G, communities, centrality)
            fold_rmses, fold_maes = [], []
            for fold_i in range(k_folds):
                test_idx = folds[fold_i]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != fold_i])
                train_triples = ratings[train_idx]
                test_triples = ratings[test_idx]

                model = ConSVD(n_users, n_items, len(communities), membership, alpha,
                                d=d, lr=0.01, reg=0.05, n_epochs=n_epochs, seed=fold_i)
                model.fit(train_triples, verbose=False)
                rmse, mae = model.evaluate(test_triples)
                fold_rmses.append(rmse)
                fold_maes.append(mae)
                print(f"  fold {fold_i+1}/{k_folds}: RMSE={rmse:.4f}  MAE={mae:.4f}")

            avg_rmse = float(np.mean(fold_rmses))
            avg_mae = float(np.mean(fold_maes))
            std_rmse = float(np.std(fold_rmses))
            std_mae = float(np.std(fold_maes))
            print(f"  -> AVG RMSE={avg_rmse:.4f} (+-{std_rmse:.4f})  "
                  f"AVG MAE={avg_mae:.4f} (+-{std_mae:.4f})")
            results.append({
                "cd_method": cd_name,
                "centrality": centrality,
                "avg_rmse": avg_rmse,
                "std_rmse": std_rmse,
                "avg_mae": avg_mae,
                "std_mae": std_mae,
                "n_communities": len(communities),
            })

    return results


if __name__ == "__main__":
    results = run_experiment_grid(
        "data/ciaodvd_movie_ratings.txt",
        "data/ciaodvd_trusts.txt",
        cd_methods=["louvain", "label_propagation", "bigclam"],
        centrality_types=["degree", "betweenness", "closeness"],
        k_folds=5,
        d=20,
        n_epochs=15,
    )
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/results.json")
