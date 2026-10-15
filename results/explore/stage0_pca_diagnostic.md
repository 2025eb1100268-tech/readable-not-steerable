# [EXPLORE] Stage 0 diagnostic: why the map PCA-512 readout underperforms the pooled latent

Exploratory. Run after Stage 0 because the PCA-512 readout scored far below the pooled readout
for every model, including the untrained control (prediction P0.2 had it within ±0.03).

**Check.** For seed 0 of four models: refit the PCA of the flattened 8×8×128 map on v1 train
rows exactly as in `explore/readouts.py`, then measure (a) how much of the 128-dimensional
spatial-mean subspace (the directions the pooled latent reads; unit vector per channel with weight
1/8 on each of the 64 positions) lies inside the top-512 principal components, and (b) what share
of the map's variance lies in that subspace.

| model (seed 0) | (a) share of the spatial-mean subspace inside the top-512 PCs | (b) share of map variance in the spatial-mean subspace |
|---|---|---|
| untrained | 0.101 | 0.399 |
| jepa_spatial | 0.488 | 0.216 |
| pixel | 0.083 | 0.023 |
| static_vicreg | 0.643 | 0.422 |

**Reading.** The top-512 PCs mostly hold position-specific structure (8192 = 64 positions × 128
channels), so the truncation discards most of the spatial-mean directions that carry the pooled
information: 90% of them for the untrained control and 92% for the pixel model. The PCA-512
readout is therefore not a superset of the pooled latent, and its low scores reflect the reduction,
not missing information in the map. The per-channel mean ⊕ std readout contains the pooled latent
exactly and is the informative map readout for the Stage 0 question.

Command (inline, logged in results/LOG.md): for each run, accumulate the map mean and centred
covariance over v1 train rows (float32 batches, float64 sum), `torch.linalg.eigh`, B = top-512
eigenvectors, U = spatial-mean directions; (a) = ‖Bᵀ U‖²_F / 128, (b) = tr(Uᵀ C U) / tr(C).
