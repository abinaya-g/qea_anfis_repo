# Findings Log

Chronological record of every validated result from development. All numbers here came from
actually executing code against the real STEW raw-signal export (`X_stew_raw.npy`,
`metadata_stew_raw.csv`) — nothing is estimated or fabricated (Rule 21).

## 1. Data verification
- `X_raw.shape = (14208, 14, 256)`, axes confirmed (epochs, channels, samples).
- 48 subjects, perfectly balanced: 296 epochs/subject, 148/class each.
- No NaN/Inf/duplicates.
- Raw values are large-offset ADC-style counts (mean ≈ 4333, range 0–8402), not µV-scale.
- Per-channel baseline offsets span ~4183 (F7) to ~4517 (F4) — a spread (~334) larger than the
  global pooled std (329) reported for the whole array.

## 2. Root cause of the reported 92.4% rejection
- Old criterion on **raw, unfiltered** data: **9.0%** rejected — does NOT reproduce 92.4% alone.
- Old criterion on **filtered** data: **91.8%** rejected — matches the reported figure closely.
- Mechanism confirmed: mean pooled epoch std drops from 274.45 (raw) to 28.57 (filtered) — a
  **9.60x** shrinkage. The fixed `z_thresh=4.0` was calibrated for the larger raw-data std and
  becomes far too aggressive once filtering shrinks the denominator.
- Ruled out as separate causes: filter edge-padding (`padlen` variants identical to default),
  DC-offset-as-direct-artifact (`demeaned` variant identical to default). `method='gust'` reduced
  rejection to 77.9%, showing edge-transient handling matters somewhat but isn't the primary driver.
- `n_bad_channels` breakdown (full dataset, filtered): 7202 epochs with 0 contaminated channels,
  3393 with exactly 1, 3613 with >1 (up to 14/14 in some epochs) — a mix of genuine pooling-bug
  false positives (1-channel cases) and epochs with real multi-channel issues.

## 3. Final QC method
- Method C (per-channel robust MAD, z_thresh=5.0, ≤2 bad channels): **16.24%** rejected on the
  full 14,208 epochs → **11,900 retained**, all 48 subjects kept, class balance 6110 LOW / 5790 HIGH.
- Cross-validated: AutoReject pilot (n=1000) gave **19.0%** rejected — independent method, close
  agreement.
- Per-subject retention: minimum 50.3% (subject 36), no subject below 50% → no exclusions needed.
- Method B (p2p, 99th percentile, ≤2 bad channels): only 0.96% rejected — too lenient by
  construction, not used as the final method.
- 50 Hz notch: max relative difference 0.55% (mean), 2.12% (max) over 50 sampled epochs vs.
  bandpass-only → confirmed redundant, dropped (`APPLY_NOTCH = False`).

## 4. Feature extraction
- 243 features per epoch (168 spectral + 3 ratios + 16 asymmetry + 42 Hjorth + 14 FuzzyEn).
- FuzzyEn unit test: sine=0.2471, white_noise=1.3713, constant=0.0000, uniform_random=1.1600 —
  passed (sine < white noise).
- `fuzzy_entropy` original implementation: 0.296s per epoch (14 channels) — bottleneck.
  `pdist`-vectorized version: 0.057s (5.2x speedup), verified numerically identical (diff < 1e-6)
  to the original on both sine and white-noise test signals before adoption.
- Full extraction: 11,900 epochs × 243 features in ~11.5–13.5 minutes (varies by run/session).

## 5. ANFIS bugs found and fixed (during baseline testing, not present in original notebook design)
- **Bug 1 (firing underflow)**: raw Gaussian products for 243-dim inputs ≈ 1e-100 to 1e-215;
  `+1e-8` normalization epsilon dominated the true sum, collapsing `predict_proba` to exactly 0.5
  for all 8000 test samples, predictions to 100% one class. Fixed with log-space softmax.
- **Bug 2 (unnecessary sigmoid)**: consequents LSE-fit directly to `{0,1}` targets, so raw output
  already ranges ≈[-0.3, 1.26]. Sigmoid centered at 0 compressed this toward 0.5. On a toy
  trivially-separable dataset: linear-threshold-at-0.5 → 1.000 accuracy; sigmoid-then-threshold →
  0.720 accuracy. Fixed by clipping raw output to [0,1] directly, no sigmoid.
- Post-fix sanity check: `n_rules=1` ANFIS matches plain OLS lstsq coefficients to 5+ decimal
  places; `n_rules=4` across 5 random premise seeds all hit 1.000 accuracy on the toy problem
  (previously 0.73–0.82 depending on seed, even with well-placed k-means-derived premise centers).

## 6. Modeling results — n_outer=5 (superseded, see n_outer=10 below)
- First full run (243 features, pop=30, gen=40): QEA-ANFIS F1=0.7315±0.046, tied with
  `standard_anfis` (F1=0.7313±0.030) but high fold variance and mixed-sign diffs (-0.037 to +0.025).
- Diagnosed as GA overfitting the inner-validation set: validation fitness climbed 8-13 points
  over 40 generations while test F1 barely moved.
- Fix: SelectKBest(f_classif, k=60) pre-filter (fit on inner-train only), generations 40→20,
  `feature_penalty_w` 0.01→0.05, `instability_w` 0.05→0.1. Result: F1=0.7286±0.028 (std nearly
  halved), mean features 30.6/243 (12.6%).
- Paired vs. baselines (n=5, Holm-corrected): no comparison significant (best raw p=0.0625 vs
  XGBoost). Effect sizes: near-zero vs logreg/MLP/standard_anfis, moderately negative vs SVM/RF/XGB.

## 7. Modeling results — n_outer=10 (final)
| Model | F1 | Balanced Acc | ROC-AUC |
|---|---|---|---|
| QEA-ANFIS (joint) | 0.7496 ± 0.048 | 0.7551 ± 0.050 | 0.836 |
| standard_anfis (A4, no GA) | 0.7556 ± 0.081 | 0.7495 ± 0.068 | 0.815 |
| logreg | 0.7578 ± 0.085 | 0.7542 ± 0.070 | 0.813 |
| mlp | 0.7441 ± 0.063 | 0.7354 ± 0.042 | 0.811 |
| random_forest | 0.7696 ± 0.053 | 0.7663 ± 0.059 | 0.852 |
| svm_rbf | 0.7670 ± 0.048 | 0.7565 ± 0.068 | 0.856 |
| xgboost | 0.7715 ± 0.051 | 0.7627 ± 0.065 | 0.857 |

Paired tests (n=10, Holm-corrected, none significant): QEA-ANFIS vs standard_anfis p=1.0000
(diff -0.0059); vs xgboost p=0.3223 (diff -0.0218); vs random_forest p=0.1934 (diff -0.0200).

Feature count: QEA-ANFIS mean 26.6/243 selected (varies 18-33 per fold).

## 8. Ablation matrix (n_outer=10, F1 mean ± std, features)
| Ablation | F1 | Bal. Acc | n_features | n_epochs | notes |
|---|---|---|---|---|---|
| A1 no artifact handling | 0.726 ± 0.071 | 0.729 | 26.8 | 14208 | |
| A2 old (buggy) z-score | 0.534 ± 0.214 (0.594 ± 0.108 excl. 1 degenerate fold) | 0.669 | 28.1 | 1077 | fold 0 had only 3 test epochs, F1=0.000 — excluded from clean estimate |
| A3/A7 quality-aware QC + joint GA (= main result) | 0.750 ± 0.048 | 0.755 | 26.6 | 11900 | |
| A4 standard ANFIS, no GA | 0.756 ± 0.081 | 0.750 | 243 | 11900 | |
| A5 feature-selection only (premises frozen) | 0.739 ± 0.068 | 0.744 | 28.8 | 11900 | |
| A6 premise-optimization only (all 60 prefiltered features) | 0.767 ± 0.053 | 0.771 | 60.0 | 11900 | |
| A8 spectral-only features | 0.732 ± 0.060 | 0.743 | 26.2 | 11900 (187 candidate feats) | |
| A9 spectral + nonlinear features | 0.748 ± 0.057 | 0.754 | 27.4 | 11900 (201 candidate feats) | |
| A10 full multi-domain | = A3/A7 | | | | |

Key paired comparisons (n_outer=10):
- Joint vs A1 (no QC): mean_diff=+0.0238, p=0.5566 — not significant, QC not shown to help or hurt.
- Joint vs A5: mean_diff=+0.0104, p=0.6953 — not significant.
- Joint vs A6: mean_diff=-0.0173, **p=0.0645** — borderline; A6 (premise-only, 60 features) may
  outperform the joint approach (30 features) by a small margin. Points-estimate-consistent
  across both n_outer=5 and n_outer=10 runs.
- A6 vs A5: mean_diff=+0.0277, p=0.0645 — same borderline signal: premise optimization
  contributes more to accuracy than feature selection does.

**Headline interpretation**: joint GA achieves A6-level accuracy (~0.75-0.77 F1) using ~half A6's
feature count (26-30 vs 60) — a real compactness-for-modest-cost trade-off, not a free lunch and
not an accuracy win over baselines. This is the most defensible novelty claim the current results
support.

## 9. Explainability
- Of 4 candidate rules on the best fold (fold 2, F1=0.835): avg firing 0.592 (Rule 3), 0.265
  (Rule 0), 0.139 (Rule 1), 0.004 (Rule 2, effectively dead).
- SHAP cross-check (independent RF, same fold's inner-train data):
  - GA-selected (27) vs top-15 SHAP: Jaccard = 0.135
  - Fuzzy-rule top features (13) vs top-15 SHAP: Jaccard = 0.120
  - GA-selected vs fuzzy-rule top features: Jaccard = 0.481 (partially circular — rule terms are
    drawn from the GA-selected set, weighted by inverse sigma)
  - GA-selected (27) vs top-27 SHAP (matched k): Jaccard = 0.227 — still modest agreement, not an
    artifact of the k-mismatch.
- Feature-selection stability (n_outer=10, top consistently-selected features): `relpow_gamma_P8`
  (8/10 folds), `logpow_theta_O2`, `relpow_theta_F4`, `relpow_alpha_O2`, `logpow_alpha_P8`,
  `logpow_gamma_F8`, `relpow_alpha_F8` (7/10 each). NOTE: this ranking differs from the n_outer=5
  run (where `hjorth_complexity_FC6` was top at 5/5) — individual feature rankings are not fully
  stable across fold-count choices; the more robust claim is at the domain level (relative/log
  alpha & theta power at occipital/frontal sites, gamma power, frontal asymmetry all recur).

## 10. Reproducibility
- Pipeline confirmed fully deterministic given fixed `RANDOM_SEED=42`: re-running
  `nested_subject_cv_v2` after a full kernel restart reproduced identical per-fold F1 values
  (0.749, 0.766, 0.704, 0.722, 0.703 at n_outer=5) to 3 decimal places.
