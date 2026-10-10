# Project — Community Detection Impact on Recommender Systems

## Setup
```
pip install networkx scikit-learn numpy scipy --break-system-packages
```

## Run
```
python3 src/evaluate.py
```
This will:
1. Load the CiaoDVD dataset (`data/ciaodvd_movie_ratings.txt`, `data/ciaodvd_trusts.txt`)
2. Run Louvain, Label Propagation, and BIGCLAM community detection once each on the trust graph
3. For each of the 3 CD methods x 3 centrality types (degree/betweenness/closeness) = 9 combinations,
   compute propensities and train/evaluate ConSVD with 5-fold cross-validation
4. Save results to `results/results.json` and print a full log to stdout

Full run takes ~10-12 minutes (betweenness/closeness centrality on the larger communities dominates
the runtime; ConSVD training itself is fast).

## Bonus: hyperparameter sensitivity study
```
python3 src/bonus_beta_sweep.py
```
Sweeps a new `beta` hyperparameter (community-term weight) for each CD method's best
centrality type, at beta = 0, 0.5, 1, 2, 5, 10, with 5-fold CV at each value. Results in
`results/beta_sweep_results.json` and `results/beta_sweep_plot.png`. See report Section 7
("Bonus: Additional Experimental Analysis") for the full write-up: what it found and why.
Takes about 15-20 minutes to run. Uses `src/consvd_beta.py`, a separate extended copy of
the model, so it does not touch or affect the main experiment (`evaluate.py`, `consvd.py`)
in any way.

## Files
- `src/data_loader.py` — loads and reindexes the CiaoDVD rating + trust data
- `src/community_detection.py` — Louvain and Label Propagation implemented FROM SCRATCH (no networkx.algorithms.community,
  no python-louvain, no third-party community-detection package -- networkx is only used for basic graph storage/traversal),
  plus BIGCLAM (NMF-based affiliation model, also from scratch)
- `src/propensity.py` — computes alpha_uc (degree/betweenness/closeness centrality within each community)
- `src/consvd.py` — the ConSVD model (community-aware SVD) with SGD training
- `src/consvd_beta.py` — bonus: ConSVD extended with a beta community-term-weight hyperparameter
- `src/evaluate.py` — orchestrates the full 5-fold CV experiment grid (main experiment)
- `src/bonus_beta_sweep.py` — bonus: runs the beta sensitivity sweep
- `results/results.json` — final RMSE/MAE table for all 9 main configurations
- `results/experiment_log.txt` — full run log for the main experiment
- `results/beta_sweep_results.json` — bonus experiment results
- `results/beta_sweep_plot.png` — bonus experiment plot (RMSE vs beta, 3 configurations)
- `data/` — the CiaoDVD dataset files used (official LibRec release)

## Note on the community detection implementations
Louvain and Label Propagation are implemented entirely by hand in `community_detection.py`
(per the course grader's requirement) -- they do not call any existing community-detection
library. As a correctness check, the from-scratch Louvain was compared against `networkx`'s
own reference Louvain on this project's trust graph: both reach the same modularity score
(0.4069), which is strong evidence the from-scratch version is correct. BIGCLAM has no
mainstream ready-made Python implementation, so it was already implemented from scratch
(via an NMF-based affiliation model) from the start of this project.

## Assumptions / deviations (see report Section 7 for full detail)
- Dataset: CiaoDVD (public LibRec release), used in place of the paper's private Ciao/Epinions crawl
- 3-of-5 CD methods implemented: Louvain, Label Propagation, BIGCLAM
- BIGCLAM fit via NMF-based affiliation model (naive gradient ascent was numerically unstable)
- alpha_uc used as a raw centrality value (no cross-community renormalization)
- ConSVD hyperparameters fixed across all 9 configurations (d=20, lr=0.01, reg=0.05, 15 epochs)
