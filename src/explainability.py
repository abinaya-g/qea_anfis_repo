"""Fuzzy rule extraction + SHAP cross-check (Rule 17).

FINDING (see docs/findings_log.md): GA/fuzzy-rule feature importance shows only MODEST overlap
with SHAP/tree-based importance -- Jaccard ~0.135 (top-15 SHAP) to ~0.227 (top-k matched to the
GA's own selection count). Report this as-is; do not round up to "agreement". A plausible
explanation is that the GA's fitness function explicitly penalizes feature-set size/redundancy,
which an unconstrained random forest has no analogous pressure toward -- so some divergence in
which features each method leans on is expected on principled grounds, not just noise.

Also note: of n_rules=4 candidate rules, only 2-3 typically carry non-trivial average firing
weight on any given fold (one rule's avg_firing is often ~0.00-0.01) -- state this plainly rather
than presenting all 4 as equally meaningful; it's a legitimate note for future work (try
n_rules=3, or add an explicit rule-pruning step).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def extract_top_rules(model, feature_names_selected, X_sample, top_k=4, top_dims=5):
    """Reports rules ranked by average firing strength on X_sample (should be TEST data, to
    reflect real generalizable behavior rather than training-set-specific firing patterns)."""
    _, firing_norm = model.forward(X_sample)
    avg_firing = firing_norm.mean(axis=0)
    top_rule_idx = np.argsort(avg_firing)[::-1][:top_k]
    rules = []
    for r in top_rule_idx:
        importance = 1 / (model.sigma[r] + 1e-6)  # sharper (lower sigma) = more discriminative
        top_d = np.argsort(importance)[::-1][:top_dims]
        desc = ", ".join([f"{feature_names_selected[d]}(mu={model.mu[r, d]:.2f})" for d in top_d])
        rules.append({'rule_idx': int(r), 'avg_firing': float(avg_firing[r]), 'top_terms': desc})
    return rules


def jaccard(set_a, set_b):
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def shap_agreement_analysis(X_train, y_train, X_test, feature_names_all, ga_selected_names,
                             rule_feature_names, top_k=None, random_state=42):
    """Trains an independent RF on the SAME fold's inner-training data (not test data) and
    compares its SHAP importance ranking against the GA's selected feature set and the fuzzy
    rules' top terms. top_k defaults to len(ga_selected_names) for a fair matched-k comparison
    (an unmatched top_k systematically caps the achievable Jaccard -- see findings_log.md)."""
    import shap
    if top_k is None:
        top_k = len(ga_selected_names)

    rf = RandomForestClassifier(n_estimators=300, random_state=random_state).fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif np.asarray(shap_values).ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_shap_idx = np.argsort(mean_abs_shap)[::-1][:top_k]
    top_shap_names = set(np.array(feature_names_all)[top_shap_idx])

    agreement = {
        'ga_vs_shap_jaccard': jaccard(set(ga_selected_names), top_shap_names),
        'rule_vs_shap_jaccard': jaccard(set(rule_feature_names), top_shap_names),
        'ga_vs_rule_jaccard': jaccard(set(ga_selected_names), set(rule_feature_names)),
    }
    return shap_values, rf, agreement, top_shap_names


def feature_selection_stability(fold_details, feature_names, top_n=20):
    """Cross-fold stability of GA-selected features (Rule 14 / Figure 9). NOTE: rankings shift
    noticeably between n_outer=5 and n_outer=10 runs on this dataset -- report the fold count
    used alongside any stability percentage, and treat individual feature rankings as noisier
    than the broader domain-level pattern (e.g. 'frontal/occipital alpha-theta power and Hjorth
    complexity keep appearing' is a more robust claim than any single feature's exact rank)."""
    from collections import Counter
    n_outer = len(fold_details)
    counts = Counter()
    for fd in fold_details:
        names_this_fold = [feature_names[i] for i in fd['selected_original_idx']]
        counts.update(names_this_fold)
    df = pd.DataFrame(counts.most_common(top_n), columns=['feature', 'folds_selected_in'])
    df['pct_of_folds'] = 100 * df['folds_selected_in'] / n_outer
    return df
