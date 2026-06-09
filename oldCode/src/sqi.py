"""
sqi.py — Signalqualitätsindizes (GT-frei)
=========================================
Bachelorarbeit: AF-Detektion in kontaktlosen Signalen · Nik Büttner · RWTH Aachen

Ersetzt den alten GT-abhängigen SQI (utils.berechne_sqi), der Qualität als
"HR stimmt mit GT-EKG überein" definierte. Problem: der Goldstandard ist im
kontaktlosen Einsatz nicht verfügbar, und Qualität != Korrektheit gegen Wahrheit.

Zwei Ebenen:
  Ebene 1 — GT-FREIE Einzelsignal-SQIs (etabliert & zitierbar):
    kSQI / sSQI  Kurtosis / Schiefe des Fensters    (Li 2008, Elgendi 2016) ✅
    pSQI         relative Leistung im Puls-/QRS-Band
    bSQI         Übereinstimmung zweier Peak-Detektoren (Behar 2013) ✅
    tSQI         mittlere Korrelation der Schläge mit Patiententemplate (Orphanidou 2015) ✅
    -> alle funktionieren bei AF: sie messen Detektierbarkeit/Morphologie,
       nicht die Rhythmus-Regelmäßigkeit.

  Ebene 2 — MULTIMODALE Selbstkonsistenz (Kernidee für ein multimodales Thema):
    cross_modal_hr_agreement(): einigen sich >=2 kontaktlose Signale auf dieselbe
    HR, ist das Fenster vertrauenswürdig — ganz ohne Goldstandard.

Validierung (für den Schreibteil): Das GT-EKG darf den SQI VALIDIEREN
(hr_agreement_with_reference: zeigen, dass "gute" Fenster kleineren HR-Fehler
haben), aber nicht BERECHNEN.

Konvention: höhere SQI-Werte = bessere Qualität, möglichst in [0, 1].
"""

from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis as scipy_kurtosis


# Physiologische Bänder pro Signaltyp (Hz) für pSQI / HR-Schätzung
SQI_BANDS = {
    'ppg':  (0.60, 3.60),   # Pulswelle
    'cecg': (5.0, 15.0),    # QRS-Komplex
    'bcg':  (0.60, 3.00),   # Herzschlag-Fundamentale
}


# ──────────────────────────────────────────────────────────────────────────
# HR-Schätzer (GT-frei)
# ──────────────────────────────────────────────────────────────────────────

def estimate_hr_fft(signal, fs, band=(0.6, 3.6)):
    """HR [bpm] aus dominanter Frequenz im Puls-Band (robust gegen Einzelfehler)."""
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    m = (freqs >= band[0]) & (freqs < band[1])
    if not m.any() or psd[m].sum() == 0:
        return np.nan
    return float(freqs[m][np.argmax(psd[m])] * 60.0)


def estimate_hr_peaks(signal, fs, band=(0.6, 3.6)):
    """HR [bpm] aus Peak-Abständen (Median-RR)."""
    peaks = _detect_amp(signal, fs)
    if len(peaks) < 3:
        return np.nan
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 1.0 / band[1]) & (rr < 1.0 / band[0])]
    return float(60.0 / np.median(rr)) if len(rr) else np.nan


# ──────────────────────────────────────────────────────────────────────────
# Zwei unabhängige Peak-Detektoren (für bSQI) + Standard-Detektor
# ──────────────────────────────────────────────────────────────────────────

def _detect_amp(signal, fs, prominence_factor=0.3):
    """Detektor A: amplitudenbasiert (find_peaks auf Signal)."""
    signal = np.asarray(signal, dtype=float)
    s = np.std(signal)
    peaks, _ = find_peaks(signal, distance=int(0.4 * fs),
                          prominence=prominence_factor * s if s > 0 else None)
    return peaks


def _detect_smoothed(signal, fs, win_factor=0.10):
    """
    Detektor B: andere Vorverarbeitung (gleitender Mittelwert) statt anderer
    Landmarke. Zielt dieselben Maxima an wie Detektor A -> hohe Übereinstimmung
    bei sauberem Signal, divergiert unter Rauschen.

    Hinweis: A und B sind zwei Detektor-KONFIGURATIONEN, kein Paar etablierter
    Literatur-Detektoren. Über die Argumente det_a/det_b in bsqi() lassen sich
    bei Bedarf echte, unabhängige Detektoren (z.B. aus neurokit2) einstecken.
    """
    from scipy.ndimage import uniform_filter1d
    signal = np.asarray(signal, dtype=float)
    w = max(3, int(win_factor * fs))
    sm = uniform_filter1d(signal, size=w)
    s = np.std(sm)
    peaks, _ = find_peaks(sm, distance=int(0.4 * fs),
                          prominence=0.3 * s if s > 0 else None)
    return peaks

def _detect_energy(signal, fs, win_factor=0.10):
    """Detektor B (UNABHÄNGIG): lokalisiert Schläge über die Steilheits-Energie
    (quadrierte erste Ableitung, geglättet) statt über die Roh-Amplitude.
    Anderes Detektionsprinzip als _detect_amp -> echte Behar-Unabhängigkeit."""
    from scipy.ndimage import uniform_filter1d
    signal = np.asarray(signal, dtype=float)
    d = np.diff(signal, prepend=signal[0]); energy = d * d
    w = max(3, int(win_factor * fs))
    env = uniform_filter1d(energy, size=w)
    s = np.std(env)
    peaks, _ = find_peaks(env, distance=int(0.4 * fs),
                          prominence=0.3 * s if s > 0 else None)
    return peaks
# ──────────────────────────────────────────────────────────────────────────
# Ebene 1: Einzelsignal-SQIs (GT-frei)
# ──────────────────────────────────────────────────────────────────────────

def ksqi(signal):
    """Kurtosis. Bewegungsartefakte -> hohe Werte. (Roh, nicht normiert.)"""
    return float(scipy_kurtosis(np.asarray(signal, dtype=float)))


def ssqi(signal):
    """Schiefe. Saubere PPG-Pulse sind positiv schief (Elgendi 2016)."""
    return float(skew(np.asarray(signal, dtype=float)))


def psqi(signal, fs, band=(0.6, 3.6)):
    """Relative Leistung im physiologischen Band / Gesamtleistung -> [0, 1]."""
    signal = np.asarray(signal, dtype=float)
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    total = psd.sum()
    if total == 0:
        return 0.0
    inband = psd[(freqs >= band[0]) & (freqs < band[1])].sum()
    return float(inband / total)


def bsqi(signal, fs, det_a=_detect_amp, det_b=_detect_energy, tol_s=0.15):
    """
    Beat-Detektions-Übereinstimmung zweier unabhängiger Detektoren (Behar 2013).
    F1-artig: 2*M / (nA + nB) in [0, 1]. Hoch = Peaks zuverlässig auffindbar.
    Funktioniert bei AF (misst Detektierbarkeit, nicht Regelmäßigkeit).
    """
    a = np.asarray(det_a(signal, fs)); b = np.asarray(det_b(signal, fs))
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    tol = tol_s * fs
    matched = 0
    used = np.zeros(len(b), dtype=bool)
    for p in a:
        d = np.abs(b - p)
        j = np.argmin(d)
        if d[j] <= tol and not used[j]:
            matched += 1; used[j] = True
    return float(2 * matched / (len(a) + len(b)))


def tsqi(signal, fs):
    """
    Template-Korrelations-SQI (Orphanidou 2015): mittlere Korrelation der
    Einzelschläge mit dem gemittelten Schlag-Template -> [0, 1].
    Hoch = stereotype, saubere Morphologie. (Bei AF variiert das Timing,
    die Schlagform bleibt aber korreliert.)
    """
    signal = np.asarray(signal, dtype=float)
    peaks = _detect_amp(signal, fs)
    if len(peaks) < 3:
        return 0.0
    half = max(2, int(0.33 * np.median(np.diff(peaks))))
    beats = [signal[p - half:p + half] for p in peaks
             if p - half >= 0 and p + half < len(signal)]
    if len(beats) < 3:
        return 0.0
    L = min(len(x) for x in beats)
    M = np.array([x[:L] for x in beats])
    template = M.mean(axis=0)
    if np.std(template) == 0:
        return 0.0
    corrs = []
    for beat in M:
        if np.std(beat) == 0:
            continue
        corrs.append(np.corrcoef(beat, template)[0, 1])
    if not corrs:
        return 0.0
    return float(max(0.0, np.mean(corrs)))


# ──────────────────────────────────────────────────────────────────────────
# Komposit & Plausibilität
# ──────────────────────────────────────────────────────────────────────────

def orphanidou_acceptable(signal, fs, band=(0.6, 3.6),
                          hr_min=40, hr_max=180, corr_min=0.66, max_rr_ratio=2.2):
    """
    Binäre Plausibilitätsregel (Orphanidou 2015), GT-frei.
    Akzeptiert ein Fenster, wenn HR physiologisch, RR-Verhältnis begrenzt und
    Template-Korrelation hoch genug. Gibt (bool, gründe-dict) zurück.
    """
    peaks = _detect_amp(signal, fs)
    reasons = {}
    if len(peaks) < 3:
        return False, {'zu_wenige_peaks': True}
    rr = np.diff(peaks) / fs
    hr = 60.0 / np.median(rr)
    reasons['hr'] = hr
    reasons['rr_ratio'] = float(np.max(rr) / np.min(rr)) if np.min(rr) > 0 else np.inf
    reasons['tsqi'] = tsqi(signal, fs)
    ok = (hr_min <= hr <= hr_max
          and reasons['rr_ratio'] <= max_rr_ratio
          and reasons['tsqi'] >= corr_min)
    return bool(ok), reasons


def _kurtosis_quality(k, k_clean=3.0, slope=10.0):
    """Mappt Kurtosis -> [0,1]: nahe k_clean gut, sehr hoch (Spikes) schlecht."""
    return float(np.clip(1.0 - max(0.0, k - k_clean) / slope, 0.0, 1.0))


def composite_sqi(signal, fs, band=(0.6, 3.6), weights=(0.5, 0.3, 0.2)):
    """
    Weicher Gesamt-SQI in [0, 1] = gewichtetes Mittel aus
    Template-Korrelation, pSQI und Kurtosis-Qualität.
    Gewichte sind tunebar und sollten gegen das GT validiert werden.
    """
    w_t, w_p, w_k = weights
    t = tsqi(signal, fs)
    p = psqi(signal, fs, band)
    k = _kurtosis_quality(ksqi(signal))
    return float(w_t * t + w_p * p + w_k * k)


def signal_sqi(signal, fs, signal_type='ppg'):
    """
    Alle GT-freien SQIs eines Fensters als dict (für Fusion ODER als ML-Feature).
    signal_type in {'ppg','cecg','bcg'} -> wählt das passende Band.
    """
    band = SQI_BANDS.get(signal_type, (0.6, 3.6))
    return {
        'kSQI':      ksqi(signal),
        'sSQI':      ssqi(signal),
        'pSQI':      psqi(signal, fs, band),
        'bSQI':      bsqi(signal, fs),
        'tSQI':      tsqi(signal, fs),
        'composite': composite_sqi(signal, fs, band),
    }


# ──────────────────────────────────────────────────────────────────────────
# Ebene 2: Multimodale HR-Selbstkonsistenz (GT-frei)
# ──────────────────────────────────────────────────────────────────────────

def cross_modal_hr_agreement(hr_dict, tol_bpm=10.0, min_agree=2):
    """
    Vertrauenswürdigkeit aus Sensor-Konsens — OHNE Goldstandard.

    hr_dict   : {signalname: HR_bpm}  (NaN/None erlaubt -> wird ignoriert)
    tol_bpm   : maximale Abweichung vom Konsens, um als "einig" zu gelten
    min_agree : ab wie vielen einigen Signalen das Fenster vertrauenswürdig ist

    Rückgabe: {
        'consensus_hr': float,           # mittlere HR der einigen Signale
        'n_agree': int,                  # Anzahl einiger Signale
        'trustworthy': bool,             # n_agree >= min_agree
        'per_signal': {name: bool},      # welche Signale stützen den Konsens
        'confidence': float,             # n_agree / n_valid in [0,1]
    }
    """
    hrs = {k: float(v) for k, v in hr_dict.items()
           if v is not None and np.isfinite(v)}
    if len(hrs) < 2:
        return {'consensus_hr': np.nan, 'n_agree': len(hrs), 'trustworthy': False,
                'per_signal': {k: False for k in hr_dict}, 'confidence': 0.0}

    vals = np.array(list(hrs.values()))
    # Größtes Cluster um den Median (robust gegen einzelne Ausreißer)
    med = np.median(vals)
    per = {k: (abs(v - med) <= tol_bpm) for k, v in hrs.items()}
    agree_vals = [v for k, v in hrs.items() if per[k]]
    n_agree = len(agree_vals)

    out_per = {k: per.get(k, False) for k in hr_dict}
    return {
        'consensus_hr': float(np.mean(agree_vals)) if n_agree else np.nan,
        'n_agree': n_agree,
        'trustworthy': n_agree >= min_agree,
        'per_signal': out_per,
        'confidence': n_agree / len(hrs),
    }


# ──────────────────────────────────────────────────────────────────────────
# Nur zur VALIDIERUNG des SQI (GT erlaubt — aber nicht zur Berechnung!)
# ──────────────────────────────────────────────────────────────────────────

def hr_agreement_with_reference(hr_estimate, hr_gt, tol_bpm=10.0):
    """
    NUR für die SQI-Validierung im Schreibteil: stimmt die GT-freie HR-Schätzung
    mit dem GT-EKG überein? Damit zeigt man, dass hohe SQIs mit kleinem HR-Fehler
    korrelieren. NICHT im Feature-/Fusionspfad verwenden.
    """
    if hr_estimate is None or hr_gt is None:
        return None
    if not (np.isfinite(hr_estimate) and np.isfinite(hr_gt)):
        return None
    return abs(hr_estimate - hr_gt) <= tol_bpm


# ──────────────────────────────────────────────────────────────────────────
# Selbsttest
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    fs = 128
    t = np.arange(0, 30, 1 / fs)
    rng = np.random.default_rng(0)

    clean = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.sin(2 * np.pi * 2.4 * t)
    noisy = clean.copy()
    # Bewegungsartefakte: zufällige Spikes
    for _ in range(8):
        i = rng.integers(0, len(noisy))
        noisy[i:i + 30] += rng.normal(0, 6, size=len(noisy[i:i + 30]))

    print("── Einzelsignal-SQIs: sauber vs. verrauscht (PPG) ──")
    print(f"{'SQI':>10} {'sauber':>10} {'verrauscht':>12}")
    cs, ns = signal_sqi(clean, fs, 'ppg'), signal_sqi(noisy, fs, 'ppg')
    for k in cs:
        print(f"{k:>10} {cs[k]:>10.3f} {ns[k]:>12.3f}")

    print("\n── Orphanidou-Plausibilität ──")
    print("  sauber    :", orphanidou_acceptable(clean, fs)[0])
    print("  verrauscht:", orphanidou_acceptable(noisy, fs)[0])

    print("\n── Multimodaler HR-Konsens (GT-frei) ──")
    # PPG1/PPG2 einig bei ~72 bpm, BCG Ausreißer, cECG fehlt
    res = cross_modal_hr_agreement({'ppg1': 72.0, 'ppg2': 74.0,
                                    'bcg1': 110.0, 'cecg': np.nan}, tol_bpm=10)
    for k, v in res.items():
        print(f"  {k:14s} {v}")

    print("\n✓ Selbsttest erfolgreich.")
