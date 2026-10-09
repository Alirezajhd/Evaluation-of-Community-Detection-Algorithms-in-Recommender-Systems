"""
bonus_beta_sweep.py
--------------------
BONUS experiment: hyperparameter sensitivity study.

Sweeps beta (the community-term weight introduced in consvd_beta.py) over
several values, for each of the 3 community detection methods (using each
method's best-performing centrality type from the main experiment:
Louvain+degree, Label Propagation+closeness, BIGCLAM+betweenness), with
5-fold cross-validation at every beta value.

Motivation: the main experiment held beta implicitly fixed at 1 and found
that RMSE/MAE barely changed across CD methods and centrality types. This
sweep tests directly whether that was because the community term's
influence was too small relative to the rest of the model, by turning its
influence up and down.
"""
import json
import numpy as np
from data_loader import build_reindexed_dataset
from community_detection import METHODS
from propensity import compute_propensities
from consvd_beta import ConSVDBeta


def k_fold_indices(n, k=5, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def run_beta_sweep(rating_path, trust_path, configs, betas, k_folds=5, d=20, n_epochs=15):
    data = build_reindexed_dataset(rating_path, trust_path)
    G = data["graph"]
    ratings = data["ratings"]
    n_users, n_items = data["n_users"], data["n_items"]
    folds = k_fold_indices(len(ratings), k=k_folds)

    cd_cache = {}
    results = []

    for cd_name, centrality in configs:
        if cd_name not in cd_cache:
            print(f"Running community detection: {cd_name} ...")
            cd_cache[cd_name] = METHODS[cd_name](G)
        communities, membership = cd_cache[cd_name]
        alpha = compute_propensities(G, communities, centrality)

        for beta in betas:
            print(f"\n=== {cd_name} + {centrality} centrality, beta={beta} ===")
            fold_rmses, fold_maes = [], []
            for fold_i in range(k_folds):
                test_idx = folds[fold_i]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != fold_i])
                train_triples = ratings[train_idx]
                test_triples = ratings[test_idx]

                model = ConSVDBeta(n_users, n_items, len(communities), membership, alpha,
                                    d=d, lr=0.01, reg=0.05, n_epochs=n_epochs, seed=fold_i, beta=beta)
                model.fit(train_triples, verbose=False)
                rmse, mae = model.evaluate(test_triples)
                fold_rmses.append(rmse)
                fold_maes.append(mae)
                print(f"  fold {fold_i+1}/{k_folds}: RMSE={rmse:.4f}  MAE={mae:.4f}")

            avg_rmse = float(np.mean(fold_rmses))
            avg_mae = float(np.mean(fold_maes))
            print(f"  -> AVG RMSE={avg_rmse:.4f}  AVG MAE={avg_mae:.4f}")
            results.append({
                "cd_method": cd_name,
                "centrality": centrality,
                "beta": beta,
                "avg_rmse": avg_rmse,
                "avg_mae": avg_mae,
            })

    return results


if __name__ == "__main__":
    configs = [
        ("louvain", "degree"),
        ("label_propagation", "closeness"),
        ("bigclam", "betweenness"),
    ]
    betas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

    results = run_beta_sweep(
        "data/ciaodvd_movie_ratings.txt",
        "data/ciaodvd_trusts.txt",
        configs=configs,
        betas=betas,
        k_folds=5,
        d=20,
        n_epochs=15,
    )
    with open("results/beta_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/beta_sweep_results.json")
