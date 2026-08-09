"""Paired statistical testing (Rule 16).

NOTE ON POWER: even at n_outer=10, most pairwise comparisons in this study do not reach p<0.05
(range roughly 0.06-1.0 -- see docs/findings_log.md for the full table). Report effect sizes
alongside p-values always, and do not describe a non-significant point-estimate difference as a
finding without that caveat attached.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def paired_stat_tests(nested_df, baseline_df, metric='f1', correction='holm'):
    """Paired at the outer-fold level. Requires nested_df and baseline_df to have been produced
    with the SAME n_outer and random_state (see cv.py) so folds correspond 1:1."""
    ganfis_scores = nested_df.sort_values('fold')[metric].values
    rows = []
    for model in baseline_df['model'].unique():
        base_scores = baseline_df[baseline_df['model'] == model].sort_values('fold')[metric].values
        if len(base_scores) != len(ganfis_scores):
            continue
        try:
            stat, p = wilcoxon(ganfis_scores, base_scores)
        except ValueError:
            stat, p = np.nan, np.nan
        diff = ganfis_scores - base_scores
        effect_size = diff.mean() / (diff.std() + 1e-12)
        rows.append({'comparison': f'QEA-ANFIS vs {model}', 'mean_diff': diff.mean(), 'stat': stat,
                      'p_raw': p, 'effect_size_d': effect_size, 'n_folds': len(ganfis_scores)})
    res_df = pd.DataFrame(rows)
    if len(res_df) and res_df['p_raw'].notna().any():
        rej, p_corr, _, _ = multipletests(res_df['p_raw'].fillna(1.0), method=correction)
        res_df['p_corrected'] = p_corr
        res_df['significant_corrected'] = rej
    return res_df


def quick_paired(df1, df2, name1, name2, metric='f1'):
    """One-off paired comparison between two already-run CV result DataFrames (e.g. two
    ablations), printed rather than returned -- convenience for exploratory checks."""
    s1 = df1.sort_values('fold')[metric].values
    s2 = df2.sort_values('fold')[metric].values
    if len(s1) != len(s2):
        print(f"{name1} vs {name2}: SKIPPED (fold count mismatch {len(s1)} vs {len(s2)})")
        return None
    diff = s1 - s2
    try:
        stat, p = wilcoxon(s1, s2)
    except ValueError:
        stat, p = np.nan, np.nan
    p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
    print(f"{name1:30s} vs {name2:25s}: mean_diff={diff.mean():+.4f}, p={p_str}")
    return {'mean_diff': diff.mean(), 'p': p}
