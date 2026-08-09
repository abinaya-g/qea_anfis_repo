"""Data loading, filtering, and artifact-QC methods (Rules 2-5).

Root-cause note (see docs/findings_log.md): the original notebook's 92.4% rejection was NOT
primarily caused by the pooled-normalization bug in isolation -- it was that bug interacting with
post-filtering variance shrinkage. Bandpass filtering cuts the pooled epoch std by ~9.6x on this
dataset, so a fixed z_thresh=4.0 that was lenient on raw data (9.0% rejection) becomes far too
aggressive on filtered data (91.8% rejection). Method C (per-channel robust MAD) fixes this by
judging each channel against its OWN median/MAD rather than a global pooled statistic, so it is
insensitive to both cross-channel baseline differences and to the raw-vs-filtered scale shift.
"""
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.stats import median_abs_deviation

FS = 128
N_CHANNELS = 14
EPOCH_SAMPLES = 256
CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']


def load_raw(data_dir):
    """Load STEW raw-signal export and verify orientation/metadata (Rule 3)."""
    X_raw = np.load(f"{data_dir}/X_stew_raw.npy")
    metadata_raw = pd.read_csv(f"{data_dir}/metadata_stew_raw.csv")

    assert X_raw.ndim == 3, f"Expected 3D array, got {X_raw.ndim}D"
    assert X_raw.shape[0] == len(metadata_raw), "epoch count vs metadata row count mismatch"
    channel_axis = [ax for ax, sz in enumerate(X_raw.shape) if sz == N_CHANNELS]
    sample_axis = [ax for ax, sz in enumerate(X_raw.shape) if sz == EPOCH_SAMPLES]
    assert channel_axis == [1] and sample_axis == [2], (
        f"X_raw layout is not (epochs, channels, samples) -- got shape {X_raw.shape}."
    )
    required_cols = {'subject', 'label'}
    missing = required_cols - set(metadata_raw.columns)
    assert not missing, f"metadata_raw missing required columns: {missing}"
    assert not np.isnan(X_raw).any() and not np.isinf(X_raw).any(), "NaN/Inf in raw data."

    return X_raw, metadata_raw


def preprocess_epoch(epoch_ch_first, fs=FS, apply_notch=False):
    """Bandpass 1-40 Hz (default no notch -- confirmed redundant given the passband, see
    docs/findings_log.md: <2.2% relative effect over 50 sampled epochs)."""
    x = epoch_ch_first
    if apply_notch:
        b_n, a_n = sig.iirnotch(50.0, 30, fs)
        x = sig.filtfilt(b_n, a_n, x, axis=1)
    nyq = fs / 2
    b, a = sig.butter(4, [1 / nyq, 40 / nyq], btype='band')
    return sig.filtfilt(b, a, x, axis=1)


def preprocess_all(X_raw, fs=FS, apply_notch=False):
    """Filter every epoch. Applied per-epoch because only pre-epoched (2 s) data is available --
    document this as a limitation (Rule 4): ideally filtering happens on continuous EEG pre-epoch."""
    return np.stack([preprocess_epoch(X_raw[i], fs, apply_notch) for i in range(X_raw.shape[0])])


def old_global_z_reject(X, z_thresh=4.0):
    """The ORIGINAL (buggy) criterion: pools mean/std over ALL channels x samples into one
    scalar per epoch. Kept here only for ablation A2 / reproducing the original bug, not for use."""
    flags = np.zeros(X.shape[0], dtype=bool)
    for i in range(X.shape[0]):
        e = X[i]
        z = np.abs((e - e.mean()) / (e.std() + 1e-8))
        flags[i] = z.max() >= z_thresh
    return flags


def method_A_none(X):
    """Control: no rejection."""
    return np.ones(X.shape[0], dtype=bool)


def method_B_p2p(X, p2p_thresh=None, max_bad_channels=2):
    """Per-channel peak-to-peak amplitude QC, data-driven 99th-percentile threshold per channel."""
    p2p = X.max(axis=2) - X.min(axis=2)
    if p2p_thresh is None:
        p2p_thresh = np.percentile(p2p, 99, axis=0)
    bad_ch = p2p > p2p_thresh[None, :]
    return bad_ch.sum(axis=1) <= max_bad_channels


def method_C_mad(X, z_thresh=5.0, max_bad_channels=2):
    """FINAL METHOD. Robust per-channel MAD-based QC: each channel judged against its own
    median/MAD, never pooled across channels. Validated: 16.24% rejection on the full 14,208
    epochs, all 48 subjects retained, cross-checked against AutoReject's independent 19.0% pilot
    estimate."""
    med = np.median(X, axis=2, keepdims=True)
    mad = median_abs_deviation(X, axis=2, scale='normal')[:, :, None] + 1e-8
    z = np.abs((X - med) / mad)
    bad_ch = z.max(axis=2) >= z_thresh
    return bad_ch.sum(axis=1) <= max_bad_channels


def method_D_autoreject(X_filt, channel_names=CHANNEL_NAMES, fs=FS, random_state=42, n_jobs=-1):
    """Optional: requires mne + autoreject. Cross-validation reference only -- Method C is the
    method actually used downstream, chosen for speed and equally good performance."""
    import mne
    from autoreject import AutoReject
    info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types='eeg')
    info.set_montage('standard_1020', match_case=False, on_missing='warn')
    epochs_mne = mne.EpochsArray(X_filt, info, verbose=False)
    ar = AutoReject(random_state=random_state, n_jobs=n_jobs, verbose=False)
    _, reject_log = ar.fit_transform(epochs_mne, return_log=True)
    return ~reject_log.bad_epochs


def qc_summary(name, keep_mask, meta):
    n_kept = int(keep_mask.sum())
    n_total = len(keep_mask)
    return {
        'method': name, 'retained': n_kept, 'rejected': n_total - n_kept,
        'rejection_pct': round(100 * (1 - n_kept / n_total), 2),
        'subjects_retained': int(meta.loc[keep_mask, 'subject'].nunique()),
        'class_balance': meta.loc[keep_mask, 'label'].value_counts().to_dict(),
    }
