"""
Signal Alignment Viewer
=======================
Visualizes the first 10 seconds of all signals for each patient,
overlaid on the GT ECG, with per-signal offset sliders.
Offsets are saved to <patient_folder>/Signals/offsets.txt on demand.

Usage:
    python signal_alignment_viewer.py <root_folder>

    <root_folder> should contain one sub-folder per patient, each with:
        Signals/New_Data.csv
        gt/NOM_ECG_ELEC_POTL_IIWaveExport.csv
        Signals/offsets.txt  (optional, pre-existing offsets)

Dependencies:
    pip install numpy scipy matplotlib
"""

import sys
import os
import csv
import json
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
import glob

# ──────────────────────────────────────────────
# Signal loading & processing (from PatientSignals)
# ──────────────────────────────────────────────

FS_DEVICE = 128   # Hz – device signals
FS_GT     = 500   # Hz – ground truth ECG
PREVIEW_S = 10    # seconds to show


def butter_bp(signal, lowcut, highcut, fs, order=4):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='bandpass')
    return filtfilt(b, a, signal)


def load_patient(folder):
    """Return dict of raw + filtered signals, or None on error."""
    signal_file = os.path.join(folder, 'Signals', 'New_Data.csv')
    gt_file     = os.path.join(folder, 'gt',
                               'NOM_ECG_ELEC_POTL_IIWaveExport.csv')

    # ── device signals ──
    try:
        with open(signal_file) as f:
            reader = csv.reader(f)
            next(reader)   # skip header
            data = np.array(list(reader), dtype=np.float64)
    except Exception as e:
        print(f"[{folder}] Cannot read Signals: {e}")
        return None

    # ── ground truth ──
    try:
        with open(gt_file) as f:
            reader = csv.reader(f)
            gt_raw = np.array(list(reader))
        gt_ecg_raw = gt_raw[:, 3].astype(np.uint16)
    except Exception as e:
        print(f"[{folder}] Cannot read GT: {e}")
        gt_ecg_raw = None

    raw = {
        'cecg' : data[:, 1],
        'ppg1' : data[:, 12],
        'ppg2' : data[:, 24],
        'bcg1' : data[:, 2],
        'bcg2' : data[:, 3],
    }

    # ── filter ──
    filt = {}
    bands = {
        'cecg': (0.5, 35,  FS_DEVICE),
        'ppg1': (0.6,  3.6, FS_DEVICE),
        'ppg2': (0.6,  3.6, FS_DEVICE),
        'bcg1': (0.6, 10,   FS_DEVICE),
        'bcg2': (0.6, 10,   FS_DEVICE),
    }
    for name, (lo, hi, fs) in bands.items():
        sig = butter_bp(raw[name], lo, hi, fs)
        if name == 'cecg' and np.median(sig) > np.mean(sig):
            sig = -sig
        filt[name] = sig

    if gt_ecg_raw is not None:
        filt['gt'] = butter_bp(gt_ecg_raw.astype(np.float64), 0.5, 35, FS_GT)
    else:
        filt['gt'] = None

    return {'raw': raw, 'filt': filt, 'n_device': data.shape[0]}


def apply_shift(signal, shift_samples):
    """
    Shift a signal by `shift_samples` (integer, can be negative).
    Positive  → signal moves right  (signal is delayed relative to GT)
    Negative  → signal moves left   (signal is advanced)
    """
    n = len(signal)
    if shift_samples == 0:
        return signal.copy()
    out = np.zeros(n)
    if shift_samples > 0:
        out[shift_samples:] = signal[:n - shift_samples]
    else:
        s = -shift_samples
        out[:n - s] = signal[s:]
    return out


def norm(sig):
    """Normalise to [-1, 1] for overlay visualisation."""
    a, b = sig.min(), sig.max()
    if b - a < 1e-12:
        return np.zeros_like(sig)
    return 2 * (sig - a) / (b - a) - 1


# ──────────────────────────────────────────────
# Offset file I/O
# ──────────────────────────────────────────────

SIGNAL_NAMES = ['cecg', 'ppg1', 'ppg2', 'bcg1', 'bcg2']

def load_offsets(folder):
    """
    Load per-signal offsets from  Signals/offsets_per_signal.json
    Returns dict {name: int_samples} with 0 defaults.
    """
    path = os.path.join(folder, 'Signals', 'offsets_per_signal.json')
    defaults = {n: 0 for n in SIGNAL_NAMES}
    if os.path.exists(path):
        try:
            with open(path) as f:
                saved = json.load(f)
            defaults.update({k: int(v) for k, v in saved.items()
                             if k in SIGNAL_NAMES})
        except Exception as e:
            print(f"Could not read offsets: {e}")
    return defaults


def save_offsets(folder, offsets):
    """
    Save per-signal offsets to  Signals/offsets_per_signal.json
    Also write a legacy offsets.txt (initial=0, final based on cecg).
    """
    path = os.path.join(folder, 'Signals', 'offsets_per_signal.json')
    with open(path, 'w') as f:
        json.dump(offsets, f, indent=2)
    print(f"Saved offsets → {path}")

    # legacy offsets.txt: store cecg offset as a simple pair
    legacy = os.path.join(folder, 'Signals', 'offsets.txt')
    cecg_off = offsets.get('cecg', 0)
    with open(legacy, 'w') as f:
        f.write(f"0\n{cecg_off}\n")
    print(f"Updated legacy  → {legacy}")


# ──────────────────────────────────────────────
# Per-patient viewer
# ──────────────────────────────────────────────

COLORS = {
    'gt'  : ('#e63946', 'GT ECG (500 Hz)'),
    'cecg': ('#457b9d', 'cECG'),
    'ppg1': ('#2a9d8f', 'PPG1'),
    'ppg2': ('#e9c46a', 'PPG2'),
    'bcg1': ('#f4a261', 'BCG1'),
    'bcg2': ('#264653', 'BCG2'),
}

SLIDER_RANGE = 256   # ± samples at device fs


def view_patient(folder, patient_data):
    filt    = patient_data['filt']
    offsets = load_offsets(folder)

    # Clip to PREVIEW_S seconds
    n_dev = min(patient_data['n_device'], PREVIEW_S * FS_DEVICE)
    n_gt  = PREVIEW_S * FS_GT if filt['gt'] is not None else 0

    t_dev = np.arange(n_dev) / FS_DEVICE
    t_gt  = np.arange(n_gt)  / FS_GT

    clips = {name: filt[name][:n_dev] for name in SIGNAL_NAMES
             if filt[name] is not None}
    gt_clip = filt['gt'][:n_gt] if filt['gt'] is not None else None

    # ── build figure ──
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"Signal Alignment — {os.path.basename(folder)}",
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[5, 1],   # plot area : button area
        hspace=0.05,
        figure=fig,
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(
        len(clips) + 1, 1,      # +1 for the GT row (always on top)
        subplot_spec=gs[0],
        hspace=0.05,
    )

    # One axis per signal so each gets its own y-scale
    axes = {}
    first_ax = None
    signal_order = ['gt'] + SIGNAL_NAMES

    for i, name in enumerate(signal_order):
        if name == 'gt' and gt_clip is None:
            continue
        if name != 'gt' and name not in clips:
            continue
        ax = fig.add_subplot(gs_top[i], sharex=first_ax)
        if first_ax is None:
            first_ax = ax
        axes[name] = ax
        color, label = COLORS[name]
        ax.set_ylabel(label, fontsize=8, color=color, rotation=0,
                      labelpad=60, va='center')
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, alpha=0.3)
        if i < len(signal_order) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel('Time (s)', fontsize=9)

    # ── draw initial lines ──
    lines = {}

    def draw_gt():
        if gt_clip is None:
            return
        ax = axes.get('gt')
        if ax is None:
            return
        if 'gt' in lines:
            lines['gt'].remove()
        color = COLORS['gt'][0]
        l, = ax.plot(t_gt, norm(gt_clip), color=color, lw=1.0, label='GT ECG')
        lines['gt'] = l
        ax.set_xlim(0, PREVIEW_S)
        ax.legend(loc='upper right', fontsize=7)

    def draw_signal(name):
        if name not in clips or name not in axes:
            return
        ax   = axes[name]
        off  = offsets[name]
        shifted = apply_shift(clips[name], off)
        if name in lines:
            lines[name].remove()
        color = COLORS[name][0]
        l, = ax.plot(t_dev, norm(shifted), color=color, lw=1.0, label=name)
        lines[name] = l
        ax.set_xlim(0, PREVIEW_S)
        ax.legend(loc='upper right', fontsize=7)

    draw_gt()
    for name in SIGNAL_NAMES:
        draw_signal(name)

    # ── sliders (one per device signal) ──
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        len(SIGNAL_NAMES), 2,
        subplot_spec=gs[1],
        hspace=0.6, wspace=0.3,
    )

    sliders = {}
    for i, name in enumerate(SIGNAL_NAMES):
        if name not in clips:
            continue
        row, col = divmod(i, 2)
        ax_sl = fig.add_subplot(gs_bot[i // 2, i % 2 + (0 if i % 2 == 0 else 0)])
        # Actually use a flat layout:
        ax_sl = fig.add_axes([
            0.10 + (i % 3) * 0.30,
            0.06 - (i // 3) * 0.04,
            0.25,
            0.018,
        ])
        color = COLORS[name][0]
        sl = Slider(
            ax_sl,
            label=name,
            valmin=-SLIDER_RANGE,
            valmax= SLIDER_RANGE,
            valinit=offsets[name],
            valstep=1,
            color=color,
        )
        sl.label.set_fontsize(8)
        sl.label.set_color(color)
        sliders[name] = sl

        def make_callback(n):
            def on_change(val):
                offsets[n] = int(val)
                draw_signal(n)
                fig.canvas.draw_idle()
            return on_change

        sl.on_changed(make_callback(name))

    # ── Save button ──
    ax_btn = fig.add_axes([0.44, 0.01, 0.12, 0.03])
    btn_save = Button(ax_btn, 'Save Offsets', color='#4caf50',
                      hovercolor='#66bb6a')
    btn_save.label.set_color('white')
    btn_save.label.set_fontsize(9)

    def on_save(_):
        save_offsets(folder, offsets)
        ax_btn.set_facecolor('#a5d6a7')
        fig.canvas.draw_idle()

    btn_save.on_clicked(on_save)

    # ── Reset button ──
    ax_rst = fig.add_axes([0.58, 0.01, 0.12, 0.03])
    btn_rst = Button(ax_rst, 'Reset All', color='#ef5350',
                     hovercolor='#e57373')
    btn_rst.label.set_color('white')
    btn_rst.label.set_fontsize(9)

    def on_reset(_):
        for name, sl in sliders.items():
            sl.set_val(0)
        ax_btn.set_facecolor('#4caf50')
        fig.canvas.draw_idle()

    btn_rst.on_clicked(on_reset)

    plt.show()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def find_patients(root):
    """
    Discover patient folders: any sub-folder that has
    Signals/New_Data.csv  AND  gt/*.csv
    """
    patients = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        has_signals = os.path.exists(os.path.join(path, 'Signals', 'New_Data.csv'))
        has_gt      = bool(glob.glob(os.path.join(path, 'gt', '*.csv')))
        if has_signals and has_gt:
            patients.append(path)
    return patients


def main():
    if len(sys.argv) < 2:
        # Try current directory as root
        root = '.'
    else:
        root = sys.argv[1]

    root = os.path.abspath(root)
    print(f"Scanning for patients in: {root}")

    # Check if root itself is a patient folder
    patients = []
    if os.path.exists(os.path.join(root, 'Signals', 'New_Data.csv')):
        patients = [root]
    else:
        patients = find_patients(root)

    if not patients:
        print("No patient folders found. Expected structure:")
        print("  <root>/")
        print("    <patient_id>/")
        print("      Signals/New_Data.csv")
        print("      gt/NOM_ECG_ELEC_POTL_IIWaveExport.csv")
        sys.exit(1)

    print(f"Found {len(patients)} patient(s):\n  " +
          "\n  ".join(os.path.basename(p) for p in patients))

    for folder in patients:
        name = os.path.basename(folder)
        print(f"\n── Loading {name} …")
        data = load_patient(folder)
        if data is None:
            print(f"   Skipping {name} (load failed)")
            continue
        print(f"   OK – {data['n_device']} device samples, "
              f"{'GT present' if data['filt']['gt'] is not None else 'no GT'}")
        view_patient(folder, data)

    print("\nDone.")


if __name__ == '__main__':
    main()
