"""GA: joint evolutionary feature selection + ANFIS premise optimization (Rules 8, 9, 10).

Chromosome = [binary feature mask] + [mu (n_rules x n_feats)] + [sigma (n_rules x n_feats)].
Consequents are NEVER part of the chromosome -- they are fit deterministically via
ANFIS.fit_consequents on training data only, inside decode_and_eval. Validation data is used
exclusively to score fitness and never touches fit_consequents (this is the fix for the original
notebook's Rule 8 bug).

Feature selection is real indexing (select_features), not elementwise masking (Rule 9) -- ANFIS
is instantiated with n_inputs=len(selected_idx), not the full feature count.

mask_fixed / premise_fixed flags exist to run ablations A5 (feature-selection only, premises
frozen at one random init) and A6 (premise-optimization only, all candidate features always
selected) -- see docs/findings_log.md for what these revealed: premise optimization does
essentially all of the accuracy work; feature selection's contribution is compactness, not
accuracy (A6 borderline-significantly outperforms A5, p=0.065 at n_outer=10).
"""
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef

from anfis import ANFIS


def select_features(X, mask):
    """Real indexing -- returns the TRUE subset (fewer columns), never zeroed columns."""
    idx = np.where(np.asarray(mask) == 1)[0]
    if len(idx) == 0:
        idx = np.array([0])
    return X[:, idx], idx


def ga_optimize(X_train, y_train, X_val, y_val, n_rules=4, pop_size=30, n_generations=20,
                 cx_prob=0.7, mut_prob=0.2, feature_penalty_w=0.05, instability_w=0.1, seed=42,
                 mask_fixed=False, premise_fixed=False):
    """Fitness (Rule 10): 0.5*F1 + 0.3*balanced_accuracy + 0.2*max(MCC,0)
                           - feature_penalty_w*(n_selected/n_total) - instability_w*bootstrap_std(F1)

    mask_fixed=True: all candidate features always selected (isolates premise optimization, A6).
    premise_fixed=True: premises frozen at one random init, never mutated (isolates feature
        selection, A5). Both False = full joint optimization (the default QEA-ANFIS config).
    """
    rng = np.random.RandomState(seed)
    n_feats = X_train.shape[1]
    n_premise = 2 * n_rules * n_feats
    fixed_premise_vec = rng.uniform(-1, 1, size=n_premise) if premise_fixed else None

    def make_individual():
        if mask_fixed:
            mask = np.ones(n_feats)
        else:
            mask = rng.randint(0, 2, size=n_feats).astype(float)
            if mask.sum() == 0:
                mask[rng.randint(n_feats)] = 1
        premise = fixed_premise_vec.copy() if premise_fixed else rng.uniform(-1, 1, size=n_premise)
        return np.concatenate([mask, premise])

    def decode_and_eval(ind, return_model=False):
        mask = (ind[:n_feats] > 0.5).astype(int)
        if mask.sum() == 0:
            return (0.0, None) if return_model else 0.0

        Xtr_sel, sel_idx = select_features(X_train, mask)
        Xv_sel, _ = select_features(X_val, mask)

        mu_full = ind[n_feats: n_feats + n_rules * n_feats].reshape(n_rules, n_feats)
        sigma_full = ind[n_feats + n_rules * n_feats: n_feats + 2 * n_rules * n_feats].reshape(n_rules, n_feats)

        model = ANFIS(n_inputs=len(sel_idx), n_rules=n_rules, random_state=seed)
        model.mu = mu_full[:, sel_idx]
        model.sigma = np.abs(sigma_full[:, sel_idx]) + 1e-3

        model.fit_consequents(Xtr_sel, y_train.astype(float))  # TRAINING data only

        preds_val = model.predict(Xv_sel)  # VALIDATION data, fitness scoring only
        f1 = f1_score(y_val, preds_val, zero_division=0)
        bal_acc = balanced_accuracy_score(y_val, preds_val)
        mcc = matthews_corrcoef(y_val, preds_val) if len(np.unique(preds_val)) > 1 else 0.0

        rng_local = np.random.RandomState(0)
        n_val = len(y_val)
        boot_f1 = [f1_score(y_val[idx_b], preds_val[idx_b], zero_division=0)
                   for idx_b in (rng_local.choice(n_val, size=n_val, replace=True) for _ in range(3))]
        instability = float(np.std(boot_f1))

        complexity_penalty = feature_penalty_w * (mask.sum() / n_feats)
        fitness = max(
            0.5 * f1 + 0.3 * bal_acc + 0.2 * max(mcc, 0) - complexity_penalty - instability_w * instability,
            0.0,
        )
        return (fitness, model) if return_model else fitness

    population = [make_individual() for _ in range(pop_size)]
    best_ind, best_fit = None, -1.0
    fitness_history = []

    for gen in range(n_generations):
        fitness = np.array([decode_and_eval(ind) for ind in population])
        gen_best_idx = fitness.argmax()
        fitness_history.append(float(fitness[gen_best_idx]))
        if fitness[gen_best_idx] > best_fit:
            best_fit = float(fitness[gen_best_idx])
            best_ind = population[gen_best_idx].copy()

        selected = []
        for _ in range(pop_size):
            i, j = rng.randint(0, pop_size, 2)
            selected.append(population[i] if fitness[i] > fitness[j] else population[j])

        next_gen = []
        for i in range(0, pop_size, 2):
            p1, p2 = selected[i], selected[(i + 1) % pop_size]
            if rng.rand() < cx_prob:
                point = rng.randint(1, len(p1))
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                if not mask_fixed and rng.rand() < mut_prob:
                    flip_idx = rng.choice(n_feats, size=max(1, n_feats // 20), replace=False)
                    c[flip_idx] = 1 - c[flip_idx]
                    if c[:n_feats].sum() == 0:
                        c[rng.randint(n_feats)] = 1
                if mask_fixed:
                    c[:n_feats] = 1
                if not premise_fixed and rng.rand() < mut_prob:
                    noise_idx = rng.choice(n_premise, size=max(1, n_premise // 20), replace=False)
                    c[n_feats + noise_idx] += rng.normal(0, 0.3, size=len(noise_idx))
                if premise_fixed:
                    c[n_feats:] = fixed_premise_vec
            next_gen.extend([c1, c2])
        population = next_gen[:pop_size]

    best_fitness, best_model = decode_and_eval(best_ind, return_model=True)
    mask_best = (best_ind[:n_feats] > 0.5).astype(int)

    return {
        'best_individual': best_ind, 'best_fitness': best_fit, 'best_model': best_model,
        'feature_mask': mask_best, 'n_selected_features': int(mask_best.sum()),
        'fitness_history': fitness_history, 'n_feats_total': n_feats,
    }
