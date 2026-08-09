# Diagnostic Report: Original `IJCIS_EEG_GA_ANFIS_pipeline.ipynb`

Rule-1 cell-by-cell diagnosis performed before any rewrite. Kept for the record / manuscript
methods section (reviewers may ask what was wrong with a naive approach).

## Structural: the notebook could not execute as-is
- `ga_optimize` (crossover logic) was truncated mid-statement (`c2 = np.conc`) — `SyntaxError`.
- `run_experiment` was also truncated (`p = clf.predict(X_te_s`) — `SyntaxError`.
- `run_experiment()` was never called anywhere; the later stats cell referenced an undefined
  global `results` — `NameError` even with syntax fixed.

## Root cause of 92.4% artifact rejection
```python
z = np.abs((epoch_2d - epoch_2d.mean()) / (epoch_2d.std() + 1e-8))
return z.max() >= z_thresh
```
`.mean()`/`.std()` with no `axis` pools ALL 14 channels × 256 samples into one scalar per epoch —
not a channel-wise criterion. Confirmed (see `findings_log.md` §2) that this bug's damage is
compounded by filtering shrinking the pooled std ~9.6x, pushing a criterion that was lenient on
raw data (9.0% rejection) to near-total rejection once applied post-filter (91.8%).

## GA-ANFIS training bug (the core methodological defect)
In `decode_and_eval`: `Xtr = X_train * mask` was computed and **never used again**. ANFIS
parameters were set directly from the GA chromosome and evaluated only against `X_val` —
training data played no role at all. The validation set was what was actually being fit against,
not held out. Feature "selection" was also implemented as elementwise masking (`X * mask`,
unselected features zeroed but still passed through the network) rather than true dimensionality
reduction — `ANFIS(n_inputs=n_feats)` was always instantiated at full width.

## Subject leakage
`run_experiment`'s inner train/val split used `train_test_split(X_tr, y_tr, ...)` at the **epoch
level**, with no group awareness — epochs from the same subject could land in both partitions.
Repeated in the "final representative model" cell used for rule-extraction/SHAP figures.

## Other issues
- `feature_names = list(metadata.columns) if X.shape[1] == len(metadata.columns) else [...]` —
  coincidental count-matching, not a real feature-name source.
- Option A's precomputed features depended on an external, unincluded
  `prepare_stew_features_patched.py` — unauditable, non-reproducible.
- 4 rules with product T-norm across up to 70+ raw inputs risked vanishing firing strengths
  (later confirmed to be a real, separate numerical-underflow bug in the rebuilt ANFIS too — see
  `findings_log.md` §5 — though the mechanism differs from what was originally hypothesized).
- Per-epoch `filtfilt` on isolated 2-second (256-sample) segments, including a 1 Hz high-pass
  edge, is questionable — safer to filter continuous EEG before epoching if available.
- Wilcoxon test on 5 outer folds, no multiple-comparison correction across 3 baseline comparisons.
- No NaN/Inf checks, no per-subject/per-channel/per-class rejection breakdown anywhere.

## What replaced each of these
See `src/` — `preprocessing.py` (QC + orientation asserts), `anfis.py` (hybrid learning fix),
`ga_optimizer.py` (real feature indexing, mask/premise separation), `cv.py` (subject-wise nested
CV with leakage asserts, identical-split baselines), `stats.py` (Holm-corrected paired tests).
