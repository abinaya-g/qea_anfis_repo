"""Nested subject-wise CV (Rules 11-12), baselines (Rule 13), metrics (Rule 15).

CRITICAL DESIGN NOTE (see docs/findings_log.md): the pre-filter (SelectKBest) inside
nested_subject_cv_v2 exists because the FIRST version without it (all 243 features in the GA
chromosome, ~2187 genes) overfit the inner-validation set badly -- validation fitness climbed 8-13
points over 40 generations while held-out test performance barely moved (classic overfitting
signature, confirmed via fitness_history inspection and fold-level paired comparison against a
no-GA baseline). Shrinking the chromosome to 60 pre-filtered candidates (fit on inner-train ONLY,
no leakage) and reducing generations from 40->20 cut F1 std roughly in half. Do not remove the
pre-filter without re-checking for the same overfitting pattern.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                              recall_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score,
                              average_precision_score, confusion_matrix)

from anfis import ANFIS
from ga_optimizer import ga_optimize


def compute_metrics(y_true, y_pred, y_prob=None):
    out = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        'kappa': cohen_kappa_score(y_true, y_pred),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out['sensitivity'] = tp / (tp + fn + 1e-12)
    out['specificity'] = tn / (tn + fp + 1e-12)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out['roc_auc'] = roc_auc_score(y_true, y_prob)
        out['pr_auc'] = average_precision_score(y_true, y_prob)
    else:
        out['roc_auc'], out['pr_auc'] = np.nan, np.nan
    return out


def nested_subject_cv_v2(X, y, groups, feature_names, n_outer=10, n_inner_val_frac=0.2, n_rules=4,
                          prefilter_k=60, ga_kwargs=None, random_state=42, verbose=True):
    """Outer StratifiedGroupKFold for test split; inner split is SUBJECT-WISE (never epoch-wise)
    for GA validation. Explicit asserts enforce zero subject overlap between every pair of
    partitions. Scaler and prefilter both fit on inner-train only (Rule 12 -- no leakage)."""
    ga_kwargs = ga_kwargs or {}
    outer_cv = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    all_results, fold_details = [], []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        train_subj_all = np.unique(groups[train_idx])
        test_subj = np.unique(groups[test_idx])
        assert len(set(train_subj_all) & set(test_subj)) == 0, "Subject leakage between outer train/test!"

        rng = np.random.RandomState(random_state + fold)
        shuffled = train_subj_all.copy()
        rng.shuffle(shuffled)
        n_val_subj = max(1, int(len(shuffled) * n_inner_val_frac))
        val_subj, inner_train_subj = shuffled[:n_val_subj], shuffled[n_val_subj:]

        inner_train_mask = np.isin(groups, inner_train_subj)
        inner_val_mask = np.isin(groups, val_subj)
        test_mask = np.isin(groups, test_subj)
        assert len(set(groups[inner_train_mask]) & set(groups[inner_val_mask])) == 0
        assert len(set(groups[inner_train_mask]) & set(groups[test_mask])) == 0
        assert len(set(groups[inner_val_mask]) & set(groups[test_mask])) == 0

        X_itr, y_itr = X[inner_train_mask], y[inner_train_mask]
        X_val, y_val = X[inner_val_mask], y[inner_val_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        scaler = StandardScaler().fit(X_itr)
        X_itr_s, X_val_s, X_te_s = scaler.transform(X_itr), scaler.transform(X_val), scaler.transform(X_te)

        selector = SelectKBest(f_classif, k=min(prefilter_k, X_itr_s.shape[1])).fit(X_itr_s, y_itr)
        prefilter_idx = selector.get_support(indices=True)
        X_itr_pf = X_itr_s[:, prefilter_idx]
        X_val_pf = X_val_s[:, prefilter_idx]
        X_te_pf = X_te_s[:, prefilter_idx]

        if verbose:
            print(f"  Fold {fold}: inner_train={X_itr.shape[0]}, inner_val={X_val.shape[0]}, "
                  f"test={X_te.shape[0]}, prefiltered to {len(prefilter_idx)} features")

        ga_result = ga_optimize(X_itr_pf, y_itr, X_val_pf, y_val, n_rules=n_rules, **ga_kwargs)
        mask = ga_result['feature_mask']
        model = ga_result['best_model']

        sel_idx_within_pf = np.where(mask == 1)[0]
        X_fit_all = np.concatenate([X_itr_pf, X_val_pf])[:, sel_idx_within_pf]
        y_fit_all = np.concatenate([y_itr, y_val]).astype(float)
        model.fit_consequents(X_fit_all, y_fit_all)

        X_te_sel = X_te_pf[:, sel_idx_within_pf]
        preds, probs = model.predict(X_te_sel), model.predict_proba(X_te_sel)

        metrics = compute_metrics(y_te, preds, probs)
        final_original_idx = prefilter_idx[sel_idx_within_pf]
        metrics.update({'fold': fold, 'n_selected_features': len(final_original_idx),
                         'test_subjects': list(test_subj)})
        all_results.append(metrics)
        if verbose:
            print(f"    -> fold {fold} test F1={metrics['f1']:.3f}, n_features={metrics['n_selected_features']}")
        fold_details.append({
            'fold': fold, 'ga_result': ga_result, 'scaler': scaler, 'prefilter_idx': prefilter_idx,
            'selected_original_idx': final_original_idx, 'model': model, 'test_idx': test_idx,
            'preds': preds, 'probs': probs, 'y_te': y_te,
        })

    return pd.DataFrame(all_results), fold_details


def run_baselines(X, y, groups, n_outer=10, random_state=42, verbose=True):
    """Identical StratifiedGroupKFold construction to nested_subject_cv_v2, required for the
    Section 15 paired statistical tests to compare like-for-like folds. Includes standard
    (non-GA) ANFIS: random premises, all features, LSE consequents on training data only."""
    outer_cv = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    models = {
        'logreg': lambda: LogisticRegression(max_iter=2000, random_state=random_state),
        'svm_rbf': lambda: SVC(kernel='rbf', probability=True, random_state=random_state),
        'random_forest': lambda: RandomForestClassifier(n_estimators=300, random_state=random_state),
        'mlp': lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=random_state),
    }
    try:
        from xgboost import XGBClassifier
        models['xgboost'] = lambda: XGBClassifier(n_estimators=300, random_state=random_state, eval_metric='logloss')
    except ImportError:
        if verbose:
            print("xgboost not installed -- skipping.")

    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
        X_tr, y_tr, X_te, y_te = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        for name, ctor in models.items():
            clf = ctor().fit(X_tr_s, y_tr)
            preds = clf.predict(X_te_s)
            probs = clf.predict_proba(X_te_s)[:, 1] if hasattr(clf, 'predict_proba') else None
            m = compute_metrics(y_te, preds, probs)
            m.update({'model': name, 'fold': fold})
            rows.append(m)

        anfis_std = ANFIS(n_inputs=X_tr_s.shape[1], n_rules=4, random_state=random_state)
        anfis_std.fit_consequents(X_tr_s, y_tr.astype(float))
        preds = anfis_std.predict(X_te_s)
        probs = anfis_std.predict_proba(X_te_s)
        m = compute_metrics(y_te, preds, probs)
        m.update({'model': 'standard_anfis', 'fold': fold})
        rows.append(m)
        if verbose:
            print(f"  Fold {fold} baselines done.")

    return pd.DataFrame(rows)
