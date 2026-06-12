import numpy as np
import csv
import json
import os
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class PatientSignals:
    """
    Loads, filters and aligns one patient's contactless signals (cECG, PPG1/2,
    BCG1/2) against the 500 Hz ground-truth ECG.

    Alignment is driven by a per-patient config file. Priority (highest first):

        1. <folder>/alignment.json     – written by the alignment viewer; SUPERSEDES all
        2. <folder>/polarity.json      – simple per-signal polarity map (±1)
        3. <folder>/Signals/offsets.json
        4. <folder>/Signals/offsets.txt

    All flips, the (drift-corrected) time offset and the start/end trim are applied
    inside offset_correction() so that a single call after filter_all() yields the
    fully aligned signals. This is intentional: the extraction pipeline only calls
    filter_all() + offset_correction(), so folding the polarity flips in here means
    they are actually applied (the old standalone apply_polarity() was never called
    by extract.py and silently had no effect).
    """

    def __init__(self, folder):
        self.folder = folder
        self.signal_file = folder + '/Signals/New_Data.csv'
        self.gt_file = folder + '/gt/NOM_ECG_ELEC_POTL_IIWaveExport.csv'
        self.data = self._read_signal_data()
        self.gt_ecg = self._read_gt_data()

        # Extract signals
        if self.data is not None:
            self.cecg = self.data[:, 1]
            self.ppg1 = self.data[:, 12]
            self.ppg2 = self.data[:, 24]
            self.bcg1 = self.data[:, 2]
            self.bcg2 = self.data[:, 3]
        else:
            self.cecg = self.ppg1 = self.ppg2 = self.bcg1 = self.bcg2 = None

        # Filtered signals (initialized as None)
        self.cecg_filt = None
        self.ppg1_filt = None
        self.ppg2_filt = None
        self.bcg1_filt = None
        self.bcg2_filt = None
        self.gt_ecg_filt = None

    # ─────────────────────────────────────────────────────────────────────
    # I/O
    # ─────────────────────────────────────────────────────────────────────

    def _read_signal_data(self):
        data = []
        try:
            with open(self.signal_file, mode='r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)   # skip header
                for row in csv_reader:
                    data.append(row)
            return np.array(data, dtype=np.float64)
        except Exception as e:
            print(f"Error reading signal data: {e}")
            return None

    def _read_gt_data(self):
        gt_data = []
        try:
            with open(self.gt_file, mode='r') as gt_file:
                csv_reader = csv.reader(gt_file)
                for row in csv_reader:
                    gt_data.append(row)
            gt_data = np.array(gt_data)
            return gt_data[:, 3].astype(np.uint16)
        except Exception as e:
            print(f"Error reading ground truth data: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────
    # Filtering
    # ─────────────────────────────────────────────────────────────────────

    def bp_filter(self, signal, lowcut, highcut, fs, order=4):
        b, a = butter(order, [lowcut, highcut], fs=fs, btype='bandpass')
        return filtfilt(b, a, signal)

    def filter_all(self, fs=128):
        """
        Band-pass each signal. Bands are kept IDENTICAL to signal_alignment_viewer.py
        on purpose: the polarity flags in alignment.json were judged by eye on the
        viewer's filtered traces, so the loader reproduces that exact band before
        applying the saved invert flags. No polarity is inferred here — flips happen
        only in offset_correction() from alignment.json.
        """
        if self.cecg is not None:
            # cECG 0.5–35 Hz — same band the viewer used when the invert flags were
            # judged by eye (NOT 0.5–20: changing the band changes the visual polarity
            # baseline the flags were set against).
            self.cecg_filt = self.bp_filter(self.cecg, 0.5, 35, fs, order=4)
            # NO heuristic auto-flip. Per project decision, the ONLY polarity flips
            # applied are the explicit ones stored in alignment.json (applied in
            # offset_correction() Step 1). Sign must be encoded there, never inferred
            # from signal statistics — otherwise flips become patient-dependent and
            # silently inconsistent.
        if self.ppg1 is not None:
            self.ppg1_filt = self.bp_filter(self.ppg1, 0.6, 3.6, fs, order=4)
        if self.ppg2 is not None:
            self.ppg2_filt = self.bp_filter(self.ppg2, 0.6, 3.6, fs, order=4)
        # BCG1 is filtered like BCG2 (0.6–10 Hz). It is deliberately NOT skipped:
        # the BCG expert and the inter-channel SQI both require bcg1_filt to be a
        # valid array, and BCG-AF information lives in the wavelet morphology of the
        # band-passed signal (see features.bcg_wavelet_feature_block), not in raw
        # broadband noise. Dropping a channel for "noise" throws away half the BCG.
        if self.bcg1 is not None:
            self.bcg1_filt = self.bp_filter(self.bcg1, 0.6, 10, fs, order=4)
        if self.bcg2 is not None:
            self.bcg2_filt = self.bp_filter(self.bcg2, 0.6, 10, fs, order=4)
        if self.gt_ecg is not None:
            self.gt_ecg_filt = self.bp_filter(self.gt_ecg, 0.5, 35, 500, order=4)

    # ─────────────────────────────────────────────────────────────────────
    # Internal alignment helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _shift_int(signal, k):
        """
        Integer-sample shift that handles both positive and negative k.

        k > 0  : delay   (pad k zeros at the front, drop k samples at the end)
        k < 0  : advance (drop |k| samples at the front, pad at the end)
        k = 0  : identity
        Matches signal_alignment_viewer.apply_shift() exactly.
        """
        n = len(signal)
        if k == 0:
            return signal.astype(float).copy()
        out = np.zeros(n, dtype=float)
        if k > 0:
            out[k:] = signal[:max(0, n - k)]
        else:
            out[:n + k] = signal[-k:]
        return out

    @staticmethod
    def _correct_signal(signal, initial_offset, final_offset):
        """
        Drift-corrected time shift (generalised: works for negative offsets too).

        Applies the integer initial_offset, then a linear drift that ramps the
        effective shift from initial_offset to final_offset across the recording
        (clock-drift between the 128 Hz device and the 500 Hz GT).

        Edge samples that fall outside the original support are filled with 0.0
        (a few samples at the recording boundary). NaN is deliberately NOT used
        here so that a tiny boundary effect cannot poison a whole 30 s window;
        explicit start/end masking is the job of the separate trim step, which
        uses NaN on purpose.
        """
        n = len(signal)
        shifted = PatientSignals._shift_int(signal, initial_offset)
        orig = np.arange(n)
        drift = np.linspace(0, final_offset - initial_offset, n)
        corrected = orig + drift
        return interp1d(corrected, shifted, bounds_error=False,
                        fill_value=0.0)(orig)

    def _load_alignment_params(self, fs):
        """
        Resolve the alignment parameters from the highest-priority file that exists
        AND parses successfully. Returns
            (flip_map, gt_flipped, initial_offset, final_offset, trim_start_s, trim_end_s)
        or None if no usable config was found.

        Unlike the original if/elif chain, a present-but-corrupt high-priority file
        falls THROUGH to the next source instead of silently disabling correction.
        """
        signal_names = ['cecg', 'ppg1', 'ppg2', 'bcg1', 'bcg2']
        flip_map = {s: False for s in signal_names}
        params = dict(flip_map=flip_map, gt_flipped=False,
                      initial_offset=0, final_offset=0,
                      trim_start_s=0.0, trim_end_s=0.0)

        alignment_path = os.path.join(self.folder, 'alignment.json')
        polarity_json  = os.path.join(self.folder, 'polarity.json')
        offsets_json   = os.path.join(self.folder, 'Signals', 'offsets.json')
        offsets_txt    = os.path.join(self.folder, 'Signals', 'offsets.txt')

        # ── 1. alignment.json (highest priority, supersedes all) ─────────────
        if os.path.exists(alignment_path):
            try:
                with open(alignment_path) as f:
                    d = json.load(f)
                inv = d.get('invert', {})
                for s in signal_names:
                    flip_map[s] = bool(inv.get(s, False))
                params.update(
                    flip_map=flip_map,
                    gt_flipped=bool(d.get('invert_gt', False)),
                    initial_offset=int(d.get('initial_offset', 0)),
                    final_offset=int(d.get('final_offset', 0)),
                    trim_start_s=float(d.get('trim_start_s', 0.0)),
                    trim_end_s=float(d.get('trim_end_s', 0.0)),
                )
                print("offset_correction: loaded alignment.json  [supersedes all]")
                print(f"  flips          = {flip_map}")
                print(f"  gt_flipped     = {params['gt_flipped']}")
                print(f"  initial_offset = {params['initial_offset']}  "
                      f"final_offset = {params['final_offset']}")
                print(f"  trim_start_s   = {params['trim_start_s']:.2f}  "
                      f"trim_end_s = {params['trim_end_s']:.2f}")
                return params
            except Exception as e:
                print(f"Warning: could not parse alignment.json: {e}  "
                      f"-> falling back to lower-priority sources")

        # ── 2. polarity.json  {"cecg": 1|-1, ..., "gt_ecg": 1|-1} ────────────
        if os.path.exists(polarity_json):
            try:
                with open(polarity_json) as f:
                    pol = json.load(f)
                for s in signal_names:
                    flip_map[s] = (int(pol.get(s, 1)) == -1)
                params['flip_map'] = flip_map
                params['gt_flipped'] = (int(pol.get('gt_ecg', 1)) == -1)
                print("offset_correction: loaded polarity.json")
                print(f"  flips      = {flip_map}")
                print(f"  gt_flipped = {params['gt_flipped']}")
                return params
            except Exception as e:
                print(f"Warning: could not parse polarity.json: {e}  -> trying next")

        # ── 3. offsets.json ──────────────────────────────────────────────────
        if os.path.exists(offsets_json):
            try:
                with open(offsets_json) as f:
                    data = json.load(f)
                params['gt_flipped'] = data.get("gt", {}).get("flipped", False)
                global_data = data.get("global", {})
                global_start = int(global_data.get("start_sample", 0))
                params['final_offset'] = int(global_data.get("time_offset_samples", 0))
                params['trim_start_s'] = global_start / fs
                if "global" not in data and "signals" in data:   # older schema
                    cecg_sig = data["signals"].get("cecg", {})
                    params['final_offset'] = int(cecg_sig.get("time_offset_samples", 0))
                    params['trim_start_s'] = int(cecg_sig.get("start_sample", 0)) / fs
                for name, vals in data.get("signals", {}).items():
                    if name in flip_map:
                        flip_map[name] = bool(vals.get("flipped", False))
                params['flip_map'] = flip_map
                print("offset_correction: loaded offsets.json")
                print(f"  flips          = {flip_map}")
                print(f"  gt_flipped     = {params['gt_flipped']}")
                print(f"  final_offset   = {params['final_offset']}  "
                      f"trim_start_s = {params['trim_start_s']:.2f}")
                return params
            except Exception as e:
                print(f"Warning: could not parse offsets.json: {e}  -> trying next")

        # ── 4. offsets.txt (last resort) ─────────────────────────────────────
        if os.path.exists(offsets_txt):
            try:
                with open(offsets_txt) as f:
                    lines = f.readlines()
                if len(lines) >= 2:
                    params['initial_offset'] = int(float(lines[0].strip()))
                    params['final_offset'] = int(float(lines[1].strip()))
                print(f"offset_correction: loaded offsets.txt  "
                      f"initial={params['initial_offset']}  final={params['final_offset']}")
                return params
            except Exception as e:
                print(f"Warning: could not parse offsets.txt: {e}")

        return None

    # ─────────────────────────────────────────────────────────────────────
    # offset_correction — flips + drift offset + trim, one call
    # ─────────────────────────────────────────────────────────────────────

    def offset_correction(self, fs=128):
        """Apply polarity flips, the drift-corrected time offset and the start/end
        trim from the highest-priority config file. Call AFTER filter_all()."""
        signal_attrs = {'cecg': 'cecg_filt', 'ppg1': 'ppg1_filt', 'ppg2': 'ppg2_filt',
                        'bcg1': 'bcg1_filt', 'bcg2': 'bcg2_filt'}

        params = self._load_alignment_params(fs)
        if params is None:
            print("offset_correction: no offset file found, skipping.")
            return

        flip_map       = params['flip_map']
        gt_flipped     = params['gt_flipped']
        initial_offset = params['initial_offset']
        final_offset   = params['final_offset']
        trim_start_s   = params['trim_start_s']
        trim_end_s     = params['trim_end_s']

        # ── Step 1: flips ────────────────────────────────────────────────
        for name, attr in signal_attrs.items():
            sig = getattr(self, attr)
            if sig is not None and flip_map.get(name, False):
                setattr(self, attr, -sig)
                print(f"  flipped {name}")
        if gt_flipped and self.gt_ecg_filt is not None:
            self.gt_ecg_filt = -self.gt_ecg_filt
            print("  flipped gt_ecg")

        # ── Step 2: drift-corrected time offset (device signals only) ────
        if initial_offset != 0 or final_offset != 0:
            for attr in signal_attrs.values():
                sig = getattr(self, attr)
                if sig is not None:
                    setattr(self, attr,
                            self._correct_signal(sig, initial_offset, final_offset))

        # ── Step 3: NaN-mask start/end trims ─────────────────────────────
        trim_start_n = int(round(trim_start_s * fs))
        trim_end_n   = int(round(trim_end_s * fs))
        if trim_start_n > 0 or trim_end_n > 0:
            for attr in signal_attrs.values():
                sig = getattr(self, attr)
                if sig is None:
                    continue
                sig = sig.astype(float).copy()
                if trim_start_n > 0:
                    sig[:min(trim_start_n, len(sig))] = np.nan
                if trim_end_n > 0:
                    sig[max(0, len(sig) - trim_end_n):] = np.nan
                setattr(self, attr, sig)
            if trim_start_n > 0:
                print(f"  start-trimmed: first {trim_start_n} samples "
                      f"({trim_start_s:.2f}s) -> NaN")
            if trim_end_n > 0:
                print(f"  end-trimmed:   last  {trim_end_n} samples "
                      f"({trim_end_s:.2f}s) -> NaN")

    def apply_polarity(self):
        """DEPRECATED. Polarity flips are now handled inside offset_correction()
        via the alignment.json / polarity.json priority chain, so they are applied
        by the standard filter_all() + offset_correction() pipeline. Kept as a
        no-op shim for backward compatibility with older notebooks."""
        print("apply_polarity() is deprecated: flips are handled in "
              "offset_correction(). No action taken.")

    # ─────────────────────────────────────────────────────────────────────
    # Plotting
    # ─────────────────────────────────────────────────────────────────────

    def plot_all_filtered(self):
        if self.data is None or self.gt_ecg_filt is None:
            print("No filtered data to plot.")
            return

        t_128 = np.arange(self.data.shape[0]) / 128.0
        t_500 = np.arange(len(self.gt_ecg_filt)) / 500.0

        fig, axs = plt.subplots(6, 1, figsize=(15, 12), sharex=True)
        axs[0].plot(t_500, self.gt_ecg_filt, label='Filtered GT ECG', color='r')
        axs[1].plot(t_128, self.cecg_filt, label='Filtered CECG')
        axs[2].plot(t_128, self.ppg1_filt, label='Filtered PPG1')
        axs[3].plot(t_128, self.ppg2_filt, label='Filtered PPG2')
        axs[4].plot(t_128, self.bcg1_filt, label='Filtered BCG1')
        axs[5].plot(t_128, self.bcg2_filt, label='Filtered BCG2')

        for ax in axs:
            ax.legend()
            ax.grid(True)
            ax.set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()
