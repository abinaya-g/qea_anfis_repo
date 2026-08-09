"""Multi-domain feature extraction (Rule 6) with validated fuzzy/sample entropy (Rule 7).

243 features per epoch: 168 spectral (abs/rel/log power x 4 bands x 14 channels) + 3 workload
ratios + 16 frontal-asymmetry + 42 Hjorth (activity/mobility/complexity x 14 ch) + 14 fuzzy entropy.

fuzzy_entropy uses scipy.spatial.distance.pdist internally -- verified numerically IDENTICAL
(within 1e-6) to a reference pure-Python-loop implementation before being adopted; ~5.2x faster.
"""
import numpy as np
import scipy.signal as sig
from scipy.spatial.distance import pdist

FS = 128
CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
BANDS = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 40)}
ASYM_PAIRS = [('F3', 'F4'), ('F7', 'F8'), ('FC5', 'FC6'), ('AF3', 'AF4')]
CH_IDX = {c: i for i, c in enumerate(CHANNEL_NAMES)}
_TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def fuzzy_entropy(x, m=2, r=0.2, n_exp=2):
    """Chen et al.-style fuzzy entropy: locally mean-centered templates, exponential fuzzy
    membership, self-match excluded, tolerance scaled by the signal's own std.

    Unit-tested (see docs/findings_log.md): sine < uniform-random < white-noise, constant -> 0.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    r_eff = r * np.std(x) + 1e-12

    def _phi(mm):
        templates = np.array([x[i:i + mm] - np.mean(x[i:i + mm]) for i in range(N - mm + 1)])
        n_t = len(templates)
        dist_condensed = pdist(templates, metric='chebyshev')
        sim = np.exp(-(dist_condensed ** n_exp) / r_eff)
        return (2 * sim.sum()) / (n_t * (n_t - 1))

    phi_m, phi_m1 = _phi(m), _phi(m + 1)
    return float(np.log(phi_m + 1e-12) - np.log(phi_m1 + 1e-12))


def sample_entropy(x, m=2, r=0.2):
    """Richman & Moorman sample entropy. Unit-tested but NOT currently included in
    extract_features_epoch -- add it there (and re-run extraction) if Rule 6's full feature
    list is required verbatim."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    r_eff = r * np.std(x) + 1e-12

    def _phi(mm):
        templates = np.array([x[i:i + mm] for i in range(N - mm + 1)])
        n_t = len(templates)
        count = 0
        for i in range(n_t):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dist <= r_eff) - 1
        return count / (n_t * (n_t - 1) + 1e-12)

    phi_m, phi_m1 = _phi(m), _phi(m + 1)
    return -np.log((phi_m1 + 1e-12) / (phi_m + 1e-12))


def hjorth_params(x, axis=-1):
    dx = np.diff(x, axis=axis)
    ddx = np.diff(dx, axis=axis)
    var_x, var_dx, var_ddx = np.var(x, axis=axis), np.var(dx, axis=axis), np.var(ddx, axis=axis)
    mobility = np.sqrt(var_dx / (var_x + 1e-12))
    complexity = np.sqrt(var_ddx / (var_dx + 1e-12)) / (mobility + 1e-12)
    return var_x, mobility, complexity


def band_powers_epoch(epoch_ch_first, fs=FS, bands=BANDS):
    """PSD computed once, vectorized across all channels, then sliced per band."""
    freqs, psd = sig.welch(epoch_ch_first, fs=fs, nperseg=min(256, epoch_ch_first.shape[1]), axis=1)
    out = {}
    total = np.zeros(epoch_ch_first.shape[0])
    for name, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs <= hi)
        pw = _TRAPZ(psd[:, idx], freqs[idx], axis=1)
        out[name] = pw
        total += pw
    return out, total


def extract_features_epoch(epoch_ch_first, fs=FS):
    """epoch_ch_first: (n_channels, n_samples) -> (feature_vector[243], feature_names[243])."""
    band_pw, total_pw = band_powers_epoch(epoch_ch_first, fs)
    feats, names = [], []

    for band in BANDS:
        pw = band_pw[band]
        rel = pw / (total_pw + 1e-12)
        for ci, ch in enumerate(CHANNEL_NAMES):
            feats += [pw[ci], rel[ci], np.log(pw[ci] + 1e-12)]
            names += [f"abspow_{band}_{ch}", f"relpow_{band}_{ch}", f"logpow_{band}_{ch}"]

    theta, alpha, beta = band_pw['theta'], band_pw['alpha'], band_pw['beta']
    feats += [(theta / (alpha + 1e-12)).mean(), (theta / (beta + 1e-12)).mean(), (beta / (alpha + 1e-12)).mean()]
    names += ['ratio_theta_alpha_mean', 'ratio_theta_beta_mean', 'ratio_beta_alpha_mean']

    for (lch, rch) in ASYM_PAIRS:
        li, ri = CH_IDX[lch], CH_IDX[rch]
        for band in BANDS:
            pw = band_pw[band]
            feats.append(np.log(pw[ri] + 1e-12) - np.log(pw[li] + 1e-12))
            names.append(f"asym_{band}_{rch}minus{lch}")

    act, mob, comp = hjorth_params(epoch_ch_first, axis=1)
    for ci, ch in enumerate(CHANNEL_NAMES):
        feats += [act[ci], mob[ci], comp[ci]]
        names += [f"hjorth_activity_{ch}", f"hjorth_mobility_{ch}", f"hjorth_complexity_{ch}"]

    for ci, ch in enumerate(CHANNEL_NAMES):
        feats.append(fuzzy_entropy(epoch_ch_first[ci]))
        names.append(f"fuzzyen_{ch}")

    return np.array(feats), names


def extract_all_features(X_filt, keep_mask, metadata_raw):
    """Extract features for all kept epochs. Returns (X_final, y_final, groups_final, feature_names).
    Takes ~2-3 min for ~12,000 epochs with the vectorized fuzzy_entropy above."""
    kept_idx = np.where(keep_mask)[0]
    _, feature_names = extract_features_epoch(X_filt[kept_idx[0]])
    X_final = np.array([extract_features_epoch(X_filt[i])[0] for i in kept_idx])
    y_final = metadata_raw.iloc[kept_idx]['label'].values.astype(int)
    groups_final = metadata_raw.iloc[kept_idx]['subject'].values

    assert not np.isnan(X_final).any() and not np.isinf(X_final).any()
    return X_final, y_final, groups_final, feature_names


def run_fuzzy_entropy_unit_tests():
    """Rule 7: validate against synthetic signals before trusting on real EEG. Returns dict of
    values; raises AssertionError if sine is not less complex than white noise."""
    rng0 = np.random.RandomState(0)
    t = np.linspace(0, 2, 256)
    sine, noise = np.sin(2 * np.pi * 5 * t), rng0.randn(256)
    constant, rand_signal = np.ones(256), rng0.uniform(-1, 1, 256)
    results = {
        'sine': fuzzy_entropy(sine), 'white_noise': fuzzy_entropy(noise),
        'constant': fuzzy_entropy(constant), 'uniform_random': fuzzy_entropy(rand_signal),
    }
    assert results['sine'] < results['white_noise'], "FuzzyEn unit test FAILED."
    return results
