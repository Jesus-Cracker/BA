"""
diagnostics.py — cECG-NaN-Diagnose & Detektor-Vergleich
=======================================================
Bachelorarbeit: AF-Detektion in kontaktlosen Signalen · Nik Büttner · RWTH Aachen

Zweck (Problem 1: 78 % NaN bei cECG-HRV):
  Findet heraus, WARUM cECG-Fenster keine HRV liefern, und vergleicht den
  heartpy-Pfad mit dem CWT-Morlet-Detektor — ohne Goldstandard, direkt auf den
  echten Fenstern.

Drei Werkzeuge:
  1. hrv_heartpy_reason(): instrumentierte Kopie von features.hrv_heartpy, die
     den AUSFALLGRUND zurückgibt (statt nur None). So sieht man, ob heartpy an
     der Schlag-Verwerfung, am Plausibilitätscheck oder an einer Exception
     scheitert.
  2. diagnose_windows(): tabelliert über viele Fenster die Erfolgs-/NaN-Raten
     beider Pfade, das Grund-Histogramm von heartpy und — wo beide Erfolg haben —
     die HR-Übereinstimmung. Liefert eine fertige Empfehlung.
  3. probe_detector(): rohe Peak-/RR-Statistik eines beliebigen Detektors
     (Peakanzahl, RR-Plausibilität) zur Parametersuche.

Zusätzlich: detect_peaks_cecg_robust() — robustere cECG-R-Peak-Variante
(Bandpass-Vorfilter + adaptive MAD-Schwelle), als Alternative zum festen
Quantil 0.80 in features.detect_peaks_cecg_cwt.

Nutzung im Notebook: Fenster als Liste von 1D-Arrays sammeln und übergeben
(siehe Selbsttest unten / Anleitung).
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from oldCode.src.features import (RR_MIN_MS, RR_MAX_MS, _rr_ms_from_detector,
                      detect_peaks_cecg_cwt, detect_peaks_simple)


# ──────────────────────────────────────────────────────────────────────────
# 1. heartpy mit Ausfallgrund
# ──────────────────────────────────────────────────────────────────────────

# Mögliche Gründe (mirror der Branches in features.hrv_heartpy):
HEARTPY_REASONS = ['ok', 'import_error', 'process_exception',
                   'nan_ibi_sdnn', 'ibi_out_of_range',
                   'no_peaks', 'too_many_rejected']


def hrv_heartpy_reason(signal, fs, reject_ratio=0.5):
    """
    Wie features.hrv_heartpy, gibt aber (meanRR_s_oder_None, grund) zurück.
    grund ∈ HEARTPY_REASONS. Für die Diagnose, NICHT für den Feature-Pfad.
    """
    try:
        import heartpy as hp
    except ImportError:
        return None, 'import_error'
    try:
        wd, m = hp.process(hp.scale_data(signal), sample_rate=fs,
                           bpmmin=30, bpmmax=220, reject_segmentwise=True)
    except Exception:
        return None, 'process_exception'
    if np.isnan(m['ibi']) or np.isnan(m['sdnn']):
        return None, 'nan_ibi_sdnn'
    if not (0.27 < m['ibi'] / 1000 < 2.0):
        return None, 'ibi_out_of_range'
    total = len(wd['peaklist']); rejected = len(wd['removed_beats'])
    if total == 0:
        return None, 'no_peaks'
    if rejected / total > reject_ratio:
        return None, 'too_many_rejected'
    return float(m['ibi'] / 1000), 'ok'


# ──────────────────────────────────────────────────────────────────────────
# 2. Robusterer cECG-Detektor (Alternative zum festen Quantil)
# ──────────────────────────────────────────────────────────────────────────

def detect_peaks_cecg_robust(signal, fs, bp=(5.0, 25.0),
                             mad_k=3.0, refractory_s=0.4, min_peaks=4):
    """
    cECG-R-Peak-Detektor mit Bandpass-Vorfilter + adaptiver MAD-Schwelle.
    Idee: QRS-Energie im 5–25-Hz-Band hervorheben, dann Peaks über
    median+mad_k·MAD (robust gegen gleichmäßiges Rauschen, kein fixes Quantil).

    bp           : Bandpass-Grenzen [Hz]
    mad_k        : Schwelle = median(env) + mad_k · MAD(env)
    refractory_s : Mindestabstand zwischen R-Peaks [s]
    """
    signal = np.asarray(signal, dtype=float)
    ny = 0.5 * fs
    lo, hi = bp[0] / ny, min(bp[1] / ny, 0.99)
    if not (0 < lo < hi < 1):
        return np.array([], dtype=int)
    try:
        b, a = butter(3, [lo, hi], btype='band')
        x = filtfilt(b, a, signal)
    except Exception:
        x = signal - np.mean(signal)
    env = x ** 2                                   # QRS-Energie-Hüllkurve
    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-12
    thr = med + mad_k * mad
    peaks, _ = find_peaks(env, height=thr, distance=max(1, int(refractory_s * fs)))
    if len(peaks) < min_peaks:                     # Fallback: lockerere Schwelle
        peaks, _ = find_peaks(env, height=med + 1.0 * mad,
                              distance=max(1, int(refractory_s * fs)))
    return peaks


# ──────────────────────────────────────────────────────────────────────────
# 3. Detektor-Sonde: rohe Peak-/RR-Statistik
# ──────────────────────────────────────────────────────────────────────────

def probe_detector(windows, fs, detector, rr_min_ms=RR_MIN_MS, rr_max_ms=RR_MAX_MS):
    """
    Wendet einen Peak-Detektor auf viele Fenster an und fasst zusammen:
    mittlere Peakzahl, Anteil Fenster <5 Peaks, Anteil Fenster mit <4
    plausiblen RR. Hilft bei der Parametersuche (Schwelle/Refraktärzeit).
    """
    n_peaks, plaus_rr, ok = [], [], 0
    for w in windows:
        try:
            pk = np.asarray(detector(w, fs))
        except Exception:
            n_peaks.append(0); plaus_rr.append(0); continue
        n_peaks.append(len(pk))
        if len(pk) >= 2:
            rr = np.diff(pk) / fs * 1000.0
            rr = rr[(rr > rr_min_ms) & (rr < rr_max_ms)]
            plaus_rr.append(len(rr))
            if len(rr) >= 4:
                ok += 1
        else:
            plaus_rr.append(0)
    n_peaks = np.array(n_peaks); plaus_rr = np.array(plaus_rr)
    n = max(1, len(windows))
    return {
        'n_fenster':          len(windows),
        'peaks_mean':         float(n_peaks.mean()),
        'peaks_median':       float(np.median(n_peaks)),
        'anteil_<5_peaks':    float(np.mean(n_peaks < 5)),
        'anteil_<4_plaus_rr': float(np.mean(plaus_rr < 4)),
        'erfolgsrate':        ok / n,
    }


# ──────────────────────────────────────────────────────────────────────────
# 4. Hauptdiagnose: heartpy vs. CWT auf denselben Fenstern
# ──────────────────────────────────────────────────────────────────────────

def diagnose_windows(windows, fs, cwt_detector=detect_peaks_cecg_cwt,
                     verbose=True):
    """
    Vergleicht für eine Liste von cECG-Fenstern (1D-Arrays):
      - heartpy-Erfolgsrate + Histogramm der Ausfallgründe
      - CWT-Erfolgsrate (liefert plausible RR?)
      - HR-Übereinstimmung, wo BEIDE Erfolg haben (|Δbpm|, Median)
    und gibt eine Empfehlung aus.

    Rückgabe: dict mit allen Kennzahlen (auch ohne pandas nutzbar).
    """
    from collections import Counter
    reasons = Counter()
    hp_ok = cwt_ok = 0
    hr_hp, hr_cwt = [], []     # bpm, nur wo der jeweilige Pfad Erfolg hat
    both_hp, both_cwt = [], []  # gepaart, wo BEIDE Erfolg

    for w in windows:
        ibi_s, reason = hrv_heartpy_reason(w, fs)
        reasons[reason] += 1
        hp_bpm = 60.0 / ibi_s if (ibi_s and ibi_s > 0) else np.nan
        if reason == 'ok':
            hp_ok += 1

        rr = _rr_ms_from_detector(w, fs, cwt_detector)
        cwt_bpm = 60000.0 / np.median(rr) if rr is not None else np.nan
        if rr is not None:
            cwt_ok += 1

        if np.isfinite(hp_bpm):
            hr_hp.append(hp_bpm)
        if np.isfinite(cwt_bpm):
            hr_cwt.append(cwt_bpm)
        if np.isfinite(hp_bpm) and np.isfinite(cwt_bpm):
            both_hp.append(hp_bpm); both_cwt.append(cwt_bpm)

    n = max(1, len(windows))
    res = {
        'n_fenster':           len(windows),
        'heartpy_erfolg':      hp_ok / n,
        'heartpy_nan':         1 - hp_ok / n,
        'cwt_erfolg':          cwt_ok / n,
        'cwt_nan':             1 - cwt_ok / n,
        'gruende':             dict(reasons),
        'n_beide_erfolg':      len(both_hp),
    }
    if both_hp:
        d = np.abs(np.array(both_hp) - np.array(both_cwt))
        res['hr_abweichung_median_bpm'] = float(np.median(d))
        res['hr_abweichung_<10bpm_anteil'] = float(np.mean(d <= 10))

    # Empfehlung — berücksichtigt, dass hohe "Erfolgsrate" ≠ korrekt ist:
    # Bei starkem Rauschen kann ein Detektor auf Rauschpeaks einrasten und
    # damit scheinbar erfolgreich sein, aber falsche HR liefern.
    disagree = res.get('hr_abweichung_median_bpm', 0.0)
    cwt_better = res['cwt_erfolg'] > res['heartpy_erfolg'] + 0.10
    if cwt_better and disagree > 15:
        rec = ("CWT liefert mehr Fenster, aber |ΔHR| Median = "
               f"{disagree:.0f} bpm: VORSICHT — CWT rastet vermutlich auf "
               "Rauschpeaks ein (peaks_mean via probe_detector prüfen; bei "
               "70 bpm/30 s sind ~35 Peaks zu erwarten). Erst Detektor "
               "härten (detect_peaks_cecg_robust, höhere mad_k) und gegen GT "
               "validieren, dann erst umstellen. Hier zeigt sich der SQI-"
               "Nutzen (Problem 2): solche Fenster gehören abgewertet.")
    elif cwt_better:
        rec = ("CWT liefert deutlich mehr Fenster bei guter HR-Übereinstimmung "
               "-> cECG-HRV auf hrv_cecg_cwt umstellen "
               "(HRV_FN['cecg'] = F.hrv_cecg_cwt).")
    elif res['heartpy_erfolg'] > res['cwt_erfolg'] + 0.10:
        rec = "heartpy ist hier besser -> beibehalten; CWT-Parameter prüfen."
    else:
        rec = ("Beide ähnlich. Dominanter Grund im gruende-Histogramm prüfen: "
               "'too_many_rejected' -> reject_ratio lockern/Vorfilter; "
               "'ibi_out_of_range'/'nan_ibi_sdnn' -> Detektion greift Rauschen.")
    res['empfehlung'] = rec

    if verbose:
        print(f"Fenster: {res['n_fenster']}")
        print(f"  heartpy Erfolg : {res['heartpy_erfolg']*100:5.1f} %  "
              f"(NaN {res['heartpy_nan']*100:.1f} %)")
        print(f"  CWT     Erfolg : {res['cwt_erfolg']*100:5.1f} %  "
              f"(NaN {res['cwt_nan']*100:.1f} %)")
        print("  heartpy-Ausfallgründe:")
        for g, c in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"     {g:18s} {c:5d}  ({c/n*100:.1f} %)")
        if both_hp:
            print(f"  wo beide Erfolg ({len(both_hp)}): |ΔHR| Median = "
                  f"{res['hr_abweichung_median_bpm']:.1f} bpm, "
                  f"<10 bpm bei {res['hr_abweichung_<10bpm_anteil']*100:.0f} %")
        print(f"  → {rec}")
    return res


# ──────────────────────────────────────────────────────────────────────────
# Synthetische cECG-Fenster für den Selbsttest
# ──────────────────────────────────────────────────────────────────────────

def _synth_ecg_window(fs, dur_s=30, bpm=75, noise=0.05, wander=0.0, seed=0):
    """Grobes EKG-Fenster: schmale QRS-Gauss-Spikes + optional Rauschen/Drift."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur_s, 1 / fs)
    rr = 60.0 / bpm
    sig = np.zeros_like(t)
    pos = 0.5
    while pos < dur_s:
        sig += 1.0 * np.exp(-((t - pos) ** 2) / (2 * 0.012 ** 2))   # R-Zacke
        sig -= 0.15 * np.exp(-((t - pos + 0.04) ** 2) / (2 * 0.02 ** 2))  # Q
        pos += rr * (1 + rng.normal(0, 0.04))                       # leichte HRV
    if wander:
        sig += wander * np.sin(2 * np.pi * 0.25 * t)
    sig += noise * rng.standard_normal(len(t))
    return sig

# ──────────────────────────────────────────────────────────────────────────
# 5. Pan–Tompkins (1985) — klassischer EKG-Referenzdetektor
#    Bandpass 5–15 Hz → Ableitung → Quadrierung → 150-ms-Integration → Schwelle.
#    Lehrbuch-Baseline neben dem CWT-Morlet-Detektor (Reviewer-Erwartung).
#    Gegen GT validieren wie CWT, dann im Detektorvergleich der Arbeit berichten.
# ──────────────────────────────────────────────────────────────────────────
def detect_peaks_pan_tompkins(signal, fs, refractory_s=0.25):
    """Classic Pan–Tompkins (1985) R-peak detector, simplified."""
    from scipy.signal import butter, filtfilt, find_peaks
    x = np.asarray(signal, float)
    ny = 0.5*fs
    b,a = butter(1, [5/ny, min(15/ny,0.99)], btype='band'); x = filtfilt(b,a,x)
    d = np.ediff1d(x, to_begin=0)                      # derivative
    sq = d*d                                           # squaring
    w = max(1, int(0.150*fs))                          # 150 ms moving integration
    mwi = np.convolve(sq, np.ones(w)/w, mode='same')
    thr = np.mean(mwi) + 0.5*np.std(mwi)
    pk,_ = find_peaks(mwi, height=thr, distance=int(refractory_s*fs))
    return pk

# ──────────────────────────────────────────────────────────────────────────
# Selbsttest
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    fs = 128

    print("── Selbsttest: saubere vs. stark+gleichmäßig verrauschte cECG-Fenster ──\n")
    clean = [_synth_ecg_window(fs, bpm=70 + 5 * (i % 4), noise=0.05, seed=i)
             for i in range(20)]
    # "stark, aber gleichmäßig verrauscht": hohe, konstante Rauschleistung
    noisy = [_synth_ecg_window(fs, bpm=70 + 5 * (i % 4), noise=0.8,
                               wander=0.3, seed=100 + i) for i in range(20)]

    print("[A] SAUBERE Fenster")
    diagnose_windows(clean, fs)

    print("\n[B] STARK+GLEICHMÄSSIG VERRAUSCHTE Fenster")
    diagnose_windows(noisy, fs)

    print("\n── probe_detector: CWT-fest vs. CWT-robust (verrauscht) ──")
    print("  CWT-fest  :", {k: round(v, 3) for k, v in
                            probe_detector(noisy, fs, detect_peaks_cecg_cwt).items()})
    print("  CWT-robust:", {k: round(v, 3) for k, v in
                            probe_detector(noisy, fs, detect_peaks_cecg_robust).items()})

    print("\n✓ Selbsttest erfolgreich.")

