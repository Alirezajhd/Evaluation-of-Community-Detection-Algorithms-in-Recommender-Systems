"""
consvd.py
---------
Implementation of the ConSVD model from Liang et al., as described in
Sec. 4 of the Ghosh & Saule paper:

    r_hat(u,i) = mu + b_u + b_i + q_i^T ( p_u + sum_{c: u in c} alpha_uc * p_c )

where:
  mu          : global rating mean (scalar)
  b_u, b_i    : user and item bias terms
  q_i         : item latent vector (d,)
  p_u         : user latent (personal preference) vector (d,)
  p_c         : community latent (rating pattern) vector (d,)
  alpha_uc    : precomputed propensity of user u in community c (fixed,
                from propensity.py -- not learned)

Trained by regularized SGD (standard SVD-style updates), minimizing:

    sum_{(u,i,r) in train} (r - r_hat(u,i))^2
      + lambda * ( b_u^2 + b_i^2 + ||q_i||^2 + ||p_u||^2 + sum_c ||p_c||^2 )
"""
import numpy as np


class ConSVD:
    def __init__(self, n_users, n_items, n_communities, membership, alpha,
                 d=20, lr=0.01, reg=0.05, n_epochs=20, seed=42):
        """
        membership: dict[user] -> set(community_id)
        alpha:      dict[(user, community_id)] -> propensity (from propensity.py)
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_communities = n_communities
        self.membership = membership
        self.alpha = alpha
        self.d = d
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

        rng = np.random.RandomState(seed)
        self.mu = 0.0
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.q = rng.normal(0, 0.1, size=(n_items, d))
        self.p_u = rng.normal(0, 0.1, size=(n_users, d))
        self.p_c = rng.normal(0, 0.1, size=(n_communities, d))

    def _user_pref_vector(self, u):
        vec = self.p_u[u].copy()
        for c in self.membership.get(u, ()):
            vec += self.alpha.get((u, c), 0.0) * self.p_c[c]
        return vec

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
                pref = self._user_pref_vector(u)
                pred = self.mu + self.b_u[u] + self.b_i[i] + self.q[i] @ pref
                err = r - pred
                sq_err_sum += err ** 2

                # bias updates
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])

                q_i_old = self.q[i].copy()
                # item vector update
                self.q[i] += self.lr * (err * pref - self.reg * self.q[i])
                # user preference vector update
                self.p_u[u] += self.lr * (err * q_i_old - self.reg * self.p_u[u])
                # community vectors update (only for communities u belongs to)
                for c in self.membership.get(u, ()):
                    a = self.alpha.get((u, c), 0.0)
                    self.p_c[c] += self.lr * (err * a * q_i_old - self.reg * self.p_c[c])

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
            pred = min(5.0, max(1.0, pred))  # clip to valid rating range
            errs.append(r - pred)
        errs = np.array(errs)
        rmse = np.sqrt((errs ** 2).mean())
        mae = np.abs(errs).mean()
        return rmse, mae
