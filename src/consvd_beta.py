"""
consvd_beta.py
--------------
BONUS / additional experiment: an extended version of ConSVD that adds a
single new hyperparameter, beta, which scales the strength of the
community-influence term:

    r_hat(u,i) = mu + b_u + b_i + q_i^T ( p_u + beta * sum_{c: u in c} alpha_uc * p_c )

beta = 1.0 reproduces the exact original ConSVD formula from consvd.py.
beta = 0.0 turns the model into a plain biased-SVD recommender with NO
community information at all (a natural ablation baseline).
beta > 1.0 lets the community term influence predictions more strongly
than the original formula allows.

Motivation: in the main experiment (see report, Section 4.1), RMSE/MAE
barely changed across community detection methods or centrality types.
One explanation offered there is that the fixed regularization/learning
rate setup never lets the community term meaningfully influence
predictions relative to the user/item bias terms. This module tests that
explanation directly: if it's correct, increasing beta should widen the
gap between different community detection methods, since the community
term becomes a bigger share of the final prediction.

This file is a self-contained copy of consvd.py with one addition (the
beta parameter), instead of a modification of the original, so that the
original main experiment (evaluate.py) is left completely untouched and
still reproducible exactly as submitted.
"""
import numpy as np


class ConSVDBeta:
    def __init__(self, n_users, n_items, n_communities, membership, alpha,
                 d=20, lr=0.01, reg=0.05, n_epochs=15, seed=42, beta=1.0):
        self.n_users = n_users
        self.n_items = n_items
        self.n_communities = n_communities
        self.membership = membership
        self.alpha = alpha
        self.d = d
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.beta = beta

        rng = np.random.RandomState(seed)
        self.mu = 0.0
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.q = rng.normal(0, 0.1, size=(n_items, d))
        self.p_u = rng.normal(0, 0.1, size=(n_users, d))
        self.p_c = rng.normal(0, 0.1, size=(n_communities, d))

    def _community_term(self, u):
        vec = np.zeros(self.d)
        for c in self.membership.get(u, ()):
            vec += self.alpha.get((u, c), 0.0) * self.p_c[c]
        return vec

    def _user_pref_vector(self, u):
        return self.p_u[u] + self.beta * self._community_term(u)

    def predict(self, u, i):
        return self.mu + self.b_u[u] + self.b_i[i] + self.q[i] @ self._user_pref_vector(u)

    def fit(self, train_triples, verbose=False):
        self.mu = train_triples[:, 2].mean()
        n = len(train_triples)
        rng = np.random.RandomState(0)
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            sq_err_sum = 0.0
            for idx in order:
                u, i, r = train_triples[idx]
                u, i = int(u), int(i)
                comm_term = self._community_term(u)
                pref = self.p_u[u] + self.beta * comm_term
                pred = self.mu + self.b_u[u] + self.b_i[i] + self.q[i] @ pref
                err = r - pred
                sq_err_sum += err ** 2

                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                q_i_old = self.q[i].copy()
                self.q[i] += self.lr * (err * pref - self.reg * self.q[i])
                self.p_u[u] += self.lr * (err * q_i_old - self.reg * self.p_u[u])
                for c in self.membership.get(u, ()):
                    a = self.alpha.get((u, c), 0.0)
                    # gradient wrt p_c gets an extra factor of beta (chain rule)
                    self.p_c[c] += self.lr * (err * self.beta * a * q_i_old - self.reg * self.p_c[c])

            if verbose:
                rmse = np.sqrt(sq_err_sum / n)
                print(f"  epoch {epoch+1}/{self.n_epochs}  train RMSE={rmse:.4f}")

    def evaluate(self, test_triples):
        errs = []
        for u, i, r in test_triples:
            u, i = int(u), int(i)
            if u >= self.n_users or i >= self.n_items:
                continue
            pred = self.predict(u, i)
            pred = min(5.0, max(1.0, pred))
            errs.append(r - pred)
        errs = np.array(errs)
        rmse = np.sqrt((errs ** 2).mean())
        mae = np.abs(errs).mean()
        return rmse, mae
