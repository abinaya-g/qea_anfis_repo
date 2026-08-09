"""ANFIS with genuine hybrid learning (Rule 8) and two numerical fixes found during validation.

BUG 1 FIXED -- firing-strength normalization underflow: raw Gaussian products for
high-dimensional inputs (e.g. 243 raw features) underflow to ~1e-100 to 1e-300. The naive
normalization `firing / (firing.sum() + 1e-8)` then has its epsilon DOMINATE the true (tiny) sum,
collapsing every prediction toward a constant. Fixed with log-space softmax normalization
(subtract max log-firing before exponentiating).

BUG 2 FIXED -- unnecessary sigmoid on an already-calibrated linear output: consequents are
least-squares-fit directly to {0,1} targets, so the raw Sugeno output is already meaningfully
scaled near [0,1]. Passing it through sigmoid(y) (centered at y=0) destroys that calibration.
Fixed by clipping the raw output to [0,1] directly instead of applying a sigmoid.

Both bugs were caught via a synthetic separable toy-data unit test (accuracy stuck at 0.72-0.82
with either bug present, jumps to 1.000 -- matching plain OLS exactly -- with both fixed). Always
run that check (see docs/findings_log.md) before trusting a modified version of this class.
"""
import numpy as np


class ANFIS:
    def __init__(self, n_inputs, n_rules=4, ridge=1e-3, random_state=None):
        rng = np.random.RandomState(random_state)
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.ridge = ridge
        self.mu = rng.uniform(-1, 1, size=(n_rules, n_inputs))
        self.sigma = rng.uniform(0.5, 1.5, size=(n_rules, n_inputs))
        self.cons = np.zeros((n_rules, n_inputs + 1))  # fit via LSE only, never set from a GA chromosome

    def _firing(self, x):
        """Log-space softmax normalization -- avoids underflow for high-dimensional products."""
        diff = x[:, None, :] - self.mu[None, :, :]
        log_mf = -0.5 * (diff / (self.sigma[None, :, :] + 1e-6)) ** 2
        log_firing = log_mf.sum(axis=2)  # product of Gaussians -> sum in log space
        log_firing_shifted = log_firing - log_firing.max(axis=1, keepdims=True)
        exp_firing = np.exp(log_firing_shifted)
        return exp_firing / (exp_firing.sum(axis=1, keepdims=True) + 1e-12)

    def fit_consequents(self, x, y_target):
        """Weighted least squares for consequent (Sugeno) parameters, premises held fixed.
        CALLER MUST ONLY PASS TRAINING DATA HERE -- this is the fix for the original notebook's
        core bug (Rule 8), where the fitness function computed X_train*mask and never used it,
        meaning ANFIS was effectively being fit on the validation set."""
        firing_norm = self._firing(x)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        N, R = firing_norm.shape
        D1 = x_aug.shape[1]
        A = np.zeros((N, R * D1))
        for r in range(R):
            A[:, r * D1:(r + 1) * D1] = firing_norm[:, r:r + 1] * x_aug
        AtA = A.T @ A + self.ridge * np.eye(A.shape[1])
        Atb = A.T @ y_target
        self.cons = np.linalg.solve(AtA, Atb).reshape(R, D1)

    def forward(self, x):
        firing_norm = self._firing(x)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        rule_out = x_aug @ self.cons.T
        return np.sum(firing_norm * rule_out, axis=1), firing_norm

    def predict_proba(self, x):
        """Raw Sugeno output is already ~[0,1]-calibrated (LSE-fit to {0,1} targets) -- clip,
        do NOT apply a sigmoid (see BUG 2 above)."""
        y, _ = self.forward(x)
        return np.clip(y, 0.0, 1.0)

    def predict(self, x, thresh=0.5):
        return (self.predict_proba(x) >= thresh).astype(int)


def run_anfis_sanity_check(random_state=42):
    """Synthetic separable-data check that caught both bugs above. Should return accuracy == 1.0
    for n_rules=1 (mathematically equivalent to plain OLS) and very close to 1.0 for n_rules=4
    across multiple random seeds. Run this after any change to ANFIS before trusting results."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(1)
    X_toy = np.vstack([rng.randn(200, 5) - 3, np.random.RandomState(2).randn(200, 5) + 3])
    y_toy = np.array([0] * 200 + [1] * 200)
    X_toy_s = StandardScaler().fit_transform(X_toy)

    results = {}
    for n_rules in (1, 4):
        model = ANFIS(n_inputs=5, n_rules=n_rules, random_state=random_state)
        model.fit_consequents(X_toy_s, y_toy.astype(float))
        preds = model.predict(X_toy_s)
        results[f'n_rules={n_rules}'] = float((preds == y_toy).mean())

    x_aug = np.concatenate([X_toy_s, np.ones((X_toy_s.shape[0], 1))], axis=1)
    theta_ols, *_ = np.linalg.lstsq(x_aug, y_toy.astype(float), rcond=None)
    pred_ols = (x_aug @ theta_ols >= 0.5).astype(int)
    results['plain_ols_reference'] = float((pred_ols == y_toy).mean())

    assert abs(results['n_rules=1'] - results['plain_ols_reference']) < 1e-6, (
        "ANFIS n_rules=1 should exactly match plain OLS -- a regression was introduced."
    )
    return results
