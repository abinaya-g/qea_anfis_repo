# QEA-ANFIS: Quality-Aware Evolutionary Adaptive Neuro-Fuzzy Inference System

Subject-independent EEG mental-workload classification on the STEW dataset, for a manuscript
intended for *International Journal of Computational Intelligence Systems* (IJCIS).

This repo is the corrected, validated implementation replacing an earlier notebook whose artifact
pipeline rejected 92.4% of epochs and whose GA-ANFIS training routine never used training data at
all (see `docs/diagnostic_report.md`). Every number this code produces has actually been executed
against the real STEW raw-signal export — nothing here is a stand-in or estimate.

## What's actually validated (see `docs/findings_log.md` for the full trail)

- **Root cause of the original 92.4% rejection**: not primarily the pooled z-score bug itself, but
  that bug interacting with post-filtering variance shrinkage (filtering cuts pooled epoch std by
  ~9.6x, so a fixed `z_thresh=4.0` becomes far more aggressive after filtering than before).
- **Fixed QC (Method C, per-channel robust MAD)**: 16.24% rejection, 11,900/14,208 epochs retained,
  all 48 subjects kept, cross-validated against an independent AutoReject run.
- **Two additional numerical bugs found and fixed in the ANFIS implementation** during baseline
  testing (not present in the original notebook's *design*, but real bugs in this rebuild that were
  caught before they contaminated any result): firing-strength underflow in high-dimensional
  products, and an unnecessary sigmoid on an already-`{0,1}`-calibrated linear output.
- **Headline result (n_outer=10 nested CV)**: QEA-ANFIS is statistically indistinguishable from
  Random Forest, XGBoost, SVM, and standard (non-GA) ANFIS on raw accuracy (all p > 0.05), while
  using a median of ~27 selected features versus 243 extracted / 60 pre-filtered candidates — a
  compactness result, not an accuracy-improvement result. Feature-selection-only (A5) underperforms
  premise-optimization-only (A6) by a borderline-significant margin (p=0.065), meaning the joint
  GA achieves A6-level accuracy at roughly half A6's feature count, not "for free."
- **Old artifact criterion, if it had been used**: F1 = 0.534±0.214 (chance-level, high variance) —
  quantifies exactly how unusable the original pipeline's output would have been.

## Repo structure

```
qea_anfis_repo/
├── README.md
├── requirements.txt
├── src/
│   ├── preprocessing.py      # filtering, artifact QC (Methods A-D)
│   ├── features.py           # multi-domain feature extraction, fuzzy/sample entropy
│   ├── anfis.py               # ANFIS class (both numerical fixes included)
│   ├── ga_optimizer.py        # GA: joint feature selection + premise optimization
│   ├── cv.py                  # nested subject-wise CV, baselines, metrics
│   ├── stats.py                # paired statistical testing
│   └── explainability.py      # rule extraction, SHAP cross-check
├── notebooks/
│   └── QEA_ANFIS_pipeline.ipynb   # end-to-end notebook wiring src/ together, Kaggle-ready
├── outputs/                   # where results CSVs/JSON land when you run the pipeline
├── figures/                   # where figures land
└── docs/
    ├── diagnostic_report.md   # Rule-1 cell-by-cell diagnosis of the ORIGINAL broken notebook
    └── findings_log.md         # chronological log of every validated finding, with numbers
```

## Quick start (Kaggle or any environment with the STEW raw export)

```python
import sys
sys.path.insert(0, 'src')

from preprocessing import load_raw, preprocess_all, method_C_mad
from features import extract_all_features
from anfis import ANFIS
from ga_optimizer import ga_optimize
from cv import nested_subject_cv_v2, run_baselines
from stats import paired_stat_tests

X_raw, metadata_raw = load_raw(DATA_DIR)
X_filt = preprocess_all(X_raw)
keep = method_C_mad(X_filt)
X_final, y_final, groups_final, feature_names = extract_all_features(X_filt, keep, metadata_raw)

nested_df, fold_details = nested_subject_cv_v2(X_final, y_final, groups_final, feature_names, n_outer=10)
baseline_df = run_baselines(X_final, y_final, groups_final, n_outer=10)
stat_results = paired_stat_tests(nested_df, baseline_df)
```

Or just open `notebooks/QEA_ANFIS_pipeline.ipynb` directly — it imports from `src/` and runs the
full pipeline end to end, cell by cell, with the same diagnostic checkpoints used throughout
development (orientation asserts, subject-leakage asserts, FuzzyEn unit tests, stop-condition
checks per Rule 25 of the original master prompt).

## Known limitations (report these, don't hide them)

- `n_outer=10` still leaves several statistical comparisons underpowered (p in the 0.06-0.32 range
  for several ablations) — point estimates are directionally consistent but not all reach
  significance. See `docs/findings_log.md` for exact numbers.
- Filtering is applied per-epoch (2 s segments), not on continuous pre-epoch EEG, because only
  already-epoched data was available in this export.
- Of 4 candidate fuzzy rules, only 2-3 carry non-trivial average firing weight on any given fold —
  worth a rule-pruning follow-up (e.g. `n_rules=3`).
- GA/fuzzy-rule feature importance shows only modest overlap with SHAP/tree-based importance
  (Jaccard ≈ 0.23 at matched k) — reported as-is, not spun as agreement.
