"""
features.py — Konsolidierte Feature-Bibliothek (Track A)
========================================================
Bachelorarbeit: AF-Detektion in kontaktlosen Signalen · Nik Büttner · RWTH Aachen

Diese Datei vereint die zuvor verstreuten Feature-Funktionen aus
`utils.py`, `features.py`, `features_ppg.py` und den Inline-Definitionen in
`02_datenverwertung.ipynb` / `04_Multimodal_ML.ipynb` an EINEM Ort.

Gelöste Probleme gegenüber dem Altstand:
  1. EINE Frequenz-Feature-Definition statt drei widersprüchlichen.
  2. EINE HRV-Mathematik (Poincaré/SDNN/RMSSD) statt dreifach kopiert.
  3. HRV-Features sind detektor-AGNOSTISCH: der Peak-Detektor wird als
     Argument übergeben → heartpy / neurokit2 / CWT mit einer Zeile tauschbar.

Konventionen:
  - Alle HRV-Zeitwerte werden in SEKUNDEN zurückgegeben (RR intern in ms).
  - Funktionen geben bei Fehlschlag None zurück (Fenster wird später verworfen).
  - `prefix` erzeugt eindeutige Spaltennamen, z.B. prefix='ppg1' -> 'ppg1_SDNN'.

Hinweis zur Reproduzierbarkeit:
  Die ursprüngliche `features_final_cwt.csv` (92 %-Ergebnis) nutzte für die
  Frequenz-Features die HRV-typischen LF/HF-Bänder (0.04-0.40 Hz), angewandt auf
  das ROHsignal. Das ist physiologisch fragwürdig (erfasst v.a. Basisliniendrift).
  Diese Datei verwendet stattdessen ein puls-orientiertes, signalgerechtes Schema.
  Wer die alte CSV exakt reproduzieren will, nutzt FREQ_BANDS_LEGACY (s.u.).
"""

from __future__ import annotations
import numpy as np
from scipy.stats import skew, kurtosis as scipy_kurtosis


def _finite(x):
    """Gibt x zurück, falls endlich, sonst NaN (verhindert inf in Features)."""
    return float(x) if np.isfinite(x) else np.nan


# ──────────────────────────────────────────────────────────────────────────
# Konfiguration
# ──────────────────────────────────────────────────────────────────────────

# Kanonisches, puls-orientiertes Frequenzschema (Standard ab jetzt).
PULSE_BAND  = (0.60, 3.60)   # Herzschlag-Band (36-216 bpm)
RESP_BAND   = (0.10, 0.60)   # Atmungs-Band
HRV_KEYS    = ['meanRR', 'SDNN', 'RMSSD', 'pNN50', 'SD1', 'SD2', 'SD1_SD2']

# Nur für exakte Reproduktion der alten features_final_cwt.csv:
FREQ_BANDS_LEGACY = {'lf': (0.04, 0.15), 'hf': (0.15, 0.40)}

# Physiologische RR-Plausibilitätsgrenzen (ms): 0.27 s = 222 bpm, 2.0 s = 30 bpm
RR_MIN_MS = 300.0
RR_MAX_MS = 2000.0


# ──────────────────────────────────────────────────────────────────────────
# 1. Zeitbereich
# ──────────────────────────────────────────────────────────────────────────

def time_domain_features(signal, prefix):
    """6 morphologische Kennwerte des Rohfensters."""
    signal = np.asarray(signal, dtype=float)
    return {
        f'{prefix}_mean':     float(np.mean(signal)),
        f'{prefix}_std':      float(np.std(signal)),
        f'{prefix}_skew':     float(skew(signal)),
        f'{prefix}_kurtosis': float(scipy_kurtosis(signal)),
        f'{prefix}_rms':      float(np.sqrt(np.mean(signal ** 2))),
        f'{prefix}_range':    float(np.max(signal) - np.min(signal)),
    }


# ──────────────────────────────────────────────────────────────────────────
# 2. Frequenzbereich  (EINE Definition, puls-orientiert)
# ──────────────────────────────────────────────────────────────────────────

def _band_power(freqs, psd, lo, hi):
    return float(np.sum(psd[(freqs >= lo) & (freqs < hi)]))


def frequency_domain_features(signal, fs, prefix,
                              pulse_band=PULSE_BAND, resp_band=RESP_BAND):
    """
    Spektrale Merkmale des Rohfensters (Periodogramm).
    Ersetzt die drei alten, widersprüchlichen Definitionen.

    Liefert: Gesamtleistung, Pulsband-Leistung (abs/rel),
             Atmungsband (rel), dominante Pulsfrequenz, spektrale Entropie.
    """
    signal = np.asarray(signal, dtype=float)
    N      = len(signal)
    freqs  = np.fft.rfftfreq(N, d=1.0 / fs)
    psd    = np.abs(np.fft.rfft(signal)) ** 2 / N

    total = _band_power(freqs, psd, 0.0, fs / 2)
    puls  = _band_power(freqs, psd, *pulse_band)
    resp  = _band_power(freqs, psd, *resp_band)

    pulse_mask = (freqs >= pulse_band[0]) & (freqs < pulse_band[1])
    if pulse_mask.any():
        dominant = float(freqs[pulse_mask][np.argmax(psd[pulse_mask])])
    else:
        dominant = np.nan

    # Spektrale Entropie (Shannon, normiert auf [0,1])
    p = psd / psd.sum() if psd.sum() > 0 else psd
    p = p[p > 0]
    spec_ent = float(-np.sum(p * np.log(p)) / np.log(len(p))) if len(p) > 1 else np.nan

    return {
        f'{prefix}_freq_puls':      puls,
        f'{prefix}_freq_puls_norm': _finite(puls / total) if total > 0 else np.nan,
        f'{prefix}_freq_resp_norm': _finite(resp / total) if total > 0 else np.nan,
        f'{prefix}_freq_puls_resp': _finite(puls / resp)  if resp  > 0 else np.nan,
        f'{prefix}_freq_dominant':  dominant,
        f'{prefix}_spec_entropy':   spec_ent,
    }


def frequency_domain_features_legacy(signal, fs, prefix):
    """Reproduktion der ALTEN LF/HF-Definition (nur für CSV-Reproduktion)."""
    signal = np.asarray(signal, dtype=float)
    N      = len(signal)
    freqs  = np.fft.rfftfreq(N, d=1.0 / fs)
    psd    = np.abs(np.fft.rfft(signal)) ** 2 / N
    lf     = _band_power(freqs, psd, *FREQ_BANDS_LEGACY['lf'])
    hf     = _band_power(freqs, psd, *FREQ_BANDS_LEGACY['hf'])
    total  = _band_power(freqs, psd, 0.003, 0.40)
    return {
        f'{prefix}_freq_lf':      lf,
        f'{prefix}_freq_hf':      hf,
        f'{prefix}_freq_lf_hf':   _finite(lf / hf)    if hf    > 0 else np.nan,
        f'{prefix}_freq_lf_norm': _finite(lf / total) if total > 0 else np.nan,
        f'{prefix}_freq_hf_norm': _finite(hf / total) if total > 0 else np.nan,
    }


# ──────────────────────────────────────────────────────────────────────────
# 3. Entropie des Rohsignals
# ──────────────────────────────────────────────────────────────────────────

def sample_entropy_signal(signal, prefix, m=2, downsample_to=500):
    """
    Sample Entropy des Rohfensters (zur Recheneffizienz auf ~500 Punkte
    heruntergetastet).

    HINWEIS: Für AF ist die Entropie der RR-INTERVALLSERIE diskriminierender
    (siehe rr_entropy_features). Diese hier beschreibt die Wellenform.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) > downsample_to:
        signal = signal[::len(signal) // downsample_to]
    N = len(signal)
    r = 0.2 * np.std(signal)
    if r == 0 or N < 10:
        return {f'{prefix}_sample_entropy': np.nan}

    def phi(mm):
        templates = np.array([signal[i:i + mm] for i in range(N - mm)])
        count = sum(
            np.sum(np.max(np.abs(templates - templates[i]), axis=1) < r) - 1
            for i in range(len(templates))
        )
        return count / (N - mm)

    B, A = phi(m), phi(m + 1)
    val = -np.log(A / B) if (B > 0 and A > 0) else np.nan
    return {f'{prefix}_sample_entropy': val}


# ──────────────────────────────────────────────────────────────────────────
# 4. HRV — eine Mathematik, austauschbarer Detektor
# ──────────────────────────────────────────────────────────────────────────

def _hrv_from_rr(rr_ms, prefix=''):
    """
    Berechnet die 7 HRV-Kennwerte aus einer RR-Serie (in ms).
    Zentrale, EINZIGE Implementierung der HRV-Mathematik.
    Rückgabe in Sekunden (außer pNN50 / SD1_SD2 = dimensionslos).
    """
    rr_ms = np.asarray(rr_ms, dtype=float)
    if len(rr_ms) < 4:
        return None

    succ   = np.diff(rr_ms)
    sdnn   = np.std(rr_ms, ddof=1)
    rmssd  = np.sqrt(np.mean(succ ** 2))
    sd1    = rmssd / np.sqrt(2)
    sd2_sq = 2 * sdnn ** 2 - rmssd ** 2 / 2      # Poincaré-Identität
    sd2    = np.sqrt(sd2_sq) if sd2_sq > 0 else np.nan

    p = f'{prefix}_' if prefix else ''
    return {
        f'{p}meanRR':  float(np.mean(rr_ms) / 1000),
        f'{p}SDNN':    float(sdnn / 1000),
        f'{p}RMSSD':   float(rmssd / 1000),
        f'{p}pNN50':   float(np.sum(np.abs(succ) > 50) / len(succ)),
        f'{p}SD1':     float(sd1 / 1000),
        f'{p}SD2':     float(sd2 / 1000) if np.isfinite(sd2) else np.nan,
        f'{p}SD1_SD2': float(sd1 / sd2)  if np.isfinite(sd2) and sd2 > 0 else np.nan,
    }


def hrv_from_detector(signal, fs, peak_detector, prefix='',
                      rr_min_ms=RR_MIN_MS, rr_max_ms=RR_MAX_MS, cv_max=None):
    """
    Generische HRV-Extraktion mit AUSTAUSCHBAREM Peak-Detektor.

    peak_detector : callable(signal, fs) -> np.ndarray[int]  (Peak-Indizes)
    cv_max        : optionaler Variationskoeffizienten-Filter (z.B. 0.20 für BCG)

    -> heartpy, neurokit2, CWT-cECG, CWT-BCG lassen sich hier einstecken.
    """
    try:
        peaks = np.asarray(peak_detector(signal, fs))
    except Exception:
        return None
    if len(peaks) < 5:
        return None

    rr_ms = np.diff(peaks) / fs * 1000.0
    rr_ms = rr_ms[(rr_ms > rr_min_ms) & (rr_ms < rr_max_ms)]
    if len(rr_ms) < 4:
        return None
    if cv_max is not None and np.std(rr_ms) / np.mean(rr_ms) > cv_max:
        return None

    return _hrv_from_rr(rr_ms, prefix)


def hrv_heartpy(signal, fs, prefix=''):
    """
    Faithful-Reproduktion des heartpy-Pfads aus utils.py (für PPG/cECG).
    Behält heartpys reject_segmentwise + Plausibilitätschecks bei.
    Benötigt `pip install heartpy`. Bei Fehlschlag -> None.
    """
    try:
        import heartpy as hp
    except ImportError:
        raise ImportError("heartpy nicht installiert: pip install heartpy")
    try:
        sig = hp.scale_data(signal)
        wd, m = hp.process(sig, sample_rate=fs, bpmmin=30, bpmmax=220,
                           reject_segmentwise=True)
        if np.isnan(m['ibi']) or np.isnan(m['sdnn']):
            return None
        if not (0.27 < m['ibi'] / 1000 < 2.0):
            return None
        rejected = len(wd['removed_beats']); total = len(wd['peaklist'])
        if total == 0 or rejected / total > 0.5:
            return None
        p = f'{prefix}_' if prefix else ''
        return {
            f'{p}meanRR':  _finite(m['ibi'] / 1000),   f'{p}SDNN':  _finite(m['sdnn'] / 1000),
            f'{p}RMSSD':   _finite(m['rmssd'] / 1000), f'{p}pNN50': _finite(m['pnn50']),
            f'{p}SD1':     _finite(m['sd1'] / 1000),   f'{p}SD2':   _finite(m['sd2'] / 1000),
            f'{p}SD1_SD2': _finite(m['sd1/sd2']),
        }
    except Exception:
        return None


# ── Mitgelieferte Peak-Detektoren zum Einstecken in hrv_from_detector ──────

def detect_peaks_simple(signal, fs, prominence_factor=0.3):
    """Einfacher amplitudenbasierter Detektor (Fallback/Baseline)."""
    from scipy.signal import find_peaks
    signal = np.asarray(signal, dtype=float)
    peaks, _ = find_peaks(signal, distance=int(0.4 * fs),
                          prominence=prominence_factor * np.std(signal))
    return peaks


def detect_peaks_bcg_cwt(signal, fs, skala=20):
    """J-Peak-Detektor für BCG via CWT-Gaus2 (Sadek & Abdulrazak 2021)."""
    import pywt
    from scipy.signal import find_peaks, resample
    sig50 = resample(signal, len(signal) * 50 // fs)
    coeffs, _ = pywt.cwt(sig50, np.arange(1, skala + 15), 'gaus2')
    peaks, _ = find_peaks(coeffs[skala - 1], distance=15)
    return peaks * fs // 50      # Indizes zurück auf Original-fs skalieren

def detect_peaks_bcg_cwt_v2(signal, fs, band_hz=(1.5, 12.0),
                            refractory_s=0.4, quantil=0.70, min_peaks=4):
    """Verbesserter BCG-J-Peak-Detektor: CWT-Gaus2-Energie über ein BAND
    (statt einer festen Skala) + adaptive Quantil-Schwelle + Refraktärzeit.
    Analog zum cECG-CWT-Detektor; unterdrückt I/K-Nebenwellen + Rauschpeaks."""
    import pywt
    from scipy.signal import find_peaks
    sig = np.asarray(signal, float); sig = sig - np.mean(sig)
    scales = np.geomspace(2, 64, num=80)
    coeffs, freqs = pywt.cwt(sig, scales, 'gaus2', 1.0 / fs)
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not mask.any():
        return np.array([], dtype=int)
    energy = np.mean(coeffs[mask, :] ** 2, axis=0)
    dist = max(1, int(refractory_s * fs))
    pk, _ = find_peaks(energy, height=np.quantile(energy, quantil), distance=dist)
    if len(pk) < min_peaks:                      # Fallback: lockerere Schwelle
        pk, _ = find_peaks(energy, height=np.quantile(energy, 0.5), distance=dist)
    return pk

def hrv_bcg_v2(signal, fs, prefix=''):
    return hrv_from_detector(signal, fs, detect_peaks_bcg_cwt_v2, prefix=prefix, cv_max=None)

def af_rr_bcg_v2(signal, fs, prefix=''):
    return af_rr_from_detector(signal, fs, detect_peaks_bcg_cwt_v2, prefix=prefix, cv_max=None)

def hrv_bcg_nogate(signal, fs, prefix=''):
    """BCG-HRV mit OLD-Detektor (detect_peaks_bcg_cwt) OHNE cv_max-Gate.
    v14 §A: bestes BCG-Setup — volle Coverage UND beste HR-Genauigkeit (~5.9 bpm)."""
    return hrv_from_detector(signal, fs, detect_peaks_bcg_cwt, prefix=prefix, cv_max=None)

def af_rr_bcg_nogate(signal, fs, prefix=''):
    return af_rr_from_detector(signal, fs, detect_peaks_bcg_cwt, prefix=prefix, cv_max=None)

def detect_peaks_cecg_cwt(signal, fs, window_s=20, quantil=0.80):
    """R-Peak-Detektor für cECG via CWT-Morlet (5-15 Hz Energieband)."""
    import pywt
    from scipy.signal import find_peaks
    win = int(window_s * fs)
    all_peaks = []
    for start in range(0, len(signal), win):
        seg = signal[start:start + win]
        if len(seg) < fs:
            continue
        scales = np.geomspace(2, 60, num=100)
        coeffs, freqs = pywt.cwt(seg, scales, 'morl', 1 / fs)
        mask = (freqs >= 5) & (freqs <= 15)
        energy = np.mean(coeffs[mask, :] ** 2, axis=0)
        pk, _ = find_peaks(energy, height=np.quantile(energy, quantil),
                           distance=int(0.4 * fs))
        all_peaks.append(pk + start)
    return np.concatenate(all_peaks) if all_peaks else np.array([], dtype=int)


# Bequeme modalitätsspezifische HRV-Wrapper (Track-A-äquivalent):
def hrv_bcg(signal, fs, prefix=''):
    return hrv_from_detector(signal, fs, detect_peaks_bcg_cwt,
                             prefix=prefix, cv_max=0.20)


def hrv_cecg_cwt(signal, fs, prefix=''):
    return hrv_from_detector(signal, fs, detect_peaks_cecg_cwt, prefix=prefix)


# ──────────────────────────────────────────────────────────────────────────
# 4b. AF-spezifische RR-Features (eigenständig, je auf einer RR-Serie in ms)
# ──────────────────────────────────────────────────────────────────────────
#
# Diese Funktionen quantifizieren die RHYTHMUS-Irregularität, die für AF
# charakteristisch ist (vs. die HRV-Kennwerte oben, die v.a. Streuung messen).
# Jede arbeitet auf einer schon extrahierten RR-Serie (in ms) und ist einzeln
# zitierbar:
#   CoSEn   Coefficient of Sample Entropy   (Lake & Moorman 2011) ✅
#   Lorenz  Lorenz-/Poincaré-Plot der δRR   (Sarkar et al. 2008) ✅
#   rrShan  Shannon-Entropie der RR-Verteilung (Dash et al. 2009) ✅
#   TPR     Turning-Point-Ratio (Zufälligkeitsmaß) (Dash et al. 2009) ✅
#   DFA-α1  kurzzeitiger DFA-Skalenexponent (Peng et al. 1995) ✅
#
# Erwartete Richtung bei AF: CoSEn↑, Lorenz-Belegung↑, rrShannon↑,
# TPR→0.667 (zufällig), DFA-α1↓ (Richtung 0.5, unkorreliert).

AF_RR_KEYS = ['CoSEn', 'rrShannon', 'TPR', 'DFA_a1',
              'Lorenz_occ', 'Lorenz_origin', 'dRR_SD']


def _sampen_counts(x, m, r):
    """Chebyshev-Template-Treffer für SampEn (Basis von CoSEn).
    Gibt (A, B) = Treffer für Längen m+1 bzw. m (ohne Selbstvergleich)."""
    x = np.asarray(x, dtype=float)
    M = len(x) - m              # gleiche Templateanzahl für m und m+1
    if M <= 1:
        return 0, 0

    def count(mm):
        T = np.array([x[i:i + mm] for i in range(M)])
        c = 0
        for i in range(M):
            d = np.max(np.abs(T - T[i]), axis=1)
            c += int(np.sum(d <= r) - 1)
        return c

    return count(m + 1), count(m)


def coefficient_sample_entropy(rr_ms, m=1, min_matches=5,
                               r0_frac=0.01, growth=1.6, max_iter=40):
    """
    CoSEn — Coefficient of Sample Entropy (Lake & Moorman 2011).
    Speziell für KURZE RR-Serien (AF-Detektion). Der Matching-Radius r wird
    so lange vergrößert, bis genügend Templatetreffer vorliegen (robust auf
    kurzen Fenstern), dann:
        CoSEn = SampEn + ln(2r) − ln(meanRR)
    Der Zusatzterm ln(2r) − ln(meanRR) = ln(2r/meanRR) ist einheitenfrei
    (r und meanRR in ms), daher ist CoSEn skaleninvariant.
    AF → höhere CoSEn. Bei Fehlschlag NaN.
    """
    rr = np.asarray(rr_ms, dtype=float)
    if len(rr) < m + 3:
        return np.nan
    mean_rr = float(np.mean(rr))
    r = max(r0_frac * float(np.std(rr)), 1.0)
    A = B = 0
    for _ in range(max_iter):
        A, B = _sampen_counts(rr, m, r)
        if B >= min_matches and A >= 1:
            break
        r *= growth
    if B < 1 or A < 1 or mean_rr <= 0:
        return np.nan
    sampen = -np.log(A / B)
    return _finite(sampen + np.log(2.0 * r) - np.log(mean_rr))


def poincare_lorenz_features(rr_ms, prefix='', cell_ms=25.0, origin_ms=80.0):
    """
    Lorenz-/Poincaré-Plot-Merkmale der δRR-Serie (Sarkar et al. 2008).
    δRR(i) = RR(i) − RR(i−1); aufgetragen wird (δRR(i), δRR(i−1)).
    Liefert:
      Lorenz_occ    Anzahl belegter Gitterzellen (cell_ms) ≈ "Irregularity
                    Evidence" — bei AF stark erhöht (weite Streuung).
      Lorenz_origin Anteil Punkte nahe Ursprung (|δRR|≤origin_ms) — regelmäßige
                    Schläge clustern dort; bei AF niedrig.
      dRR_SD        Standardabweichung von δRR [ms] — direkte Irregularität.
    """
    rr = np.asarray(rr_ms, dtype=float)
    p = f'{prefix}_' if prefix else ''
    keys = [f'{p}Lorenz_occ', f'{p}Lorenz_origin', f'{p}dRR_SD']
    if len(rr) < 4:
        return {k: np.nan for k in keys}
    drr = np.diff(rr)
    if len(drr) < 2:
        return {k: np.nan for k in keys}
    x, yv = drr[1:], drr[:-1]
    xi = np.floor(x / cell_ms).astype(int)
    yi = np.floor(yv / cell_ms).astype(int)
    occ = len(set(zip(xi.tolist(), yi.tolist())))
    near = float(np.mean((np.abs(x) <= origin_ms) & (np.abs(yv) <= origin_ms)))
    return {f'{p}Lorenz_occ': float(occ),
            f'{p}Lorenz_origin': near,
            f'{p}dRR_SD': _finite(float(np.std(drr)))}


def rr_shannon_entropy(rr_ms, bin_ms=50.0, rr_lo=RR_MIN_MS, rr_hi=RR_MAX_MS):
    """
    Shannon-Entropie der RR-Verteilung (Dash et al. 2009), normiert auf [0,1].
    WICHTIG: feste, ABSOLUTE Bins (bin_ms über [rr_lo, rr_hi]) statt
    fenster-relativer Skalierung — sonst wird eine enge Sinusverteilung
    künstlich über alle Bins gestreckt und die Trennung geht verloren.
    AF → breitere RR-Verteilung → höhere Entropie.
    """
    rr = np.asarray(rr_ms, dtype=float)
    if len(rr) < 4:
        return np.nan
    edges = np.arange(rr_lo, rr_hi + bin_ms, bin_ms)
    if len(edges) < 3:
        return np.nan
    hist, _ = np.histogram(np.clip(rr, rr_lo, rr_hi - 1e-6), bins=edges)
    if hist.sum() == 0:
        return np.nan
    pr = hist[hist > 0] / hist.sum()
    H = -np.sum(pr * np.log(pr))
    return _finite(H / np.log(len(edges) - 1))


def turning_point_ratio(rr_ms):
    """
    Turning-Point-Ratio (Dash et al. 2009): Anteil lokaler Extrema in der
    RR-Serie. Für eine ZUFÄLLIGE Serie ≈ 2/3 (0.667). AF (irregulär/zufällig)
    → nahe 0.667; glatter, korrelierter Sinusrhythmus → deutlich darunter.
    """
    rr = np.asarray(rr_ms, dtype=float)
    N = len(rr)
    if N < 3:
        return np.nan
    tp = 0
    for i in range(1, N - 1):
        a, b, c = rr[i - 1], rr[i], rr[i + 1]
        if (b > a and b > c) or (b < a and b < c):
            tp += 1
    return float(tp / (N - 2))


def dfa_alpha1(rr_ms, scale_min=4, scale_max=16):
    """
    Kurzzeit-DFA-Skalenexponent α1 (Detrended Fluctuation Analysis,
    Peng et al. 1995) über Boxgrößen [scale_min, scale_max] Schläge.
    Gesunder Sinusrhythmus: α1 ≈ 1.0–1.2 (langreichweitig korreliert).
    AF: α1 → 0.5 (unkorreliert, "weißer" RR-Verlauf). Bei Fehlschlag NaN.
    """
    rr = np.asarray(rr_ms, dtype=float)
    N = len(rr)
    if N < scale_min * 3:
        return np.nan
    smax = min(scale_max, N // 3)
    if smax < scale_min + 1:
        return np.nan
    y = np.cumsum(rr - rr.mean())
    scales, Fs = [], []
    for n in range(scale_min, smax + 1):
        nb = N // n
        if nb < 1:
            continue
        rms = []
        for b in range(nb):
            seg = y[b * n:(b + 1) * n]
            t = np.arange(n)
            fit = np.polyval(np.polyfit(t, seg, 1), t)
            rms.append(np.mean((seg - fit) ** 2))
        F = np.sqrt(np.mean(rms))
        if F > 0:
            scales.append(n); Fs.append(F)
    if len(scales) < 2:
        return np.nan
    return _finite(float(np.polyfit(np.log(scales), np.log(Fs), 1)[0]))


def rr_af_feature_block(rr_ms, prefix=''):
    """
    Bündelt alle AF-RR-Features einer RR-Serie (ms) zu einem dict.
    Schlüssel sind IMMER vorhanden (NaN bei Fehlschlag) → einheitliche
    Spaltenstruktur über alle Fenster (analog zu signal_feature_block).
    """
    p = f'{prefix}_' if prefix else ''
    rr = np.asarray(rr_ms, dtype=float) if rr_ms is not None else np.array([])
    block = {
        f'{p}CoSEn':     coefficient_sample_entropy(rr) if len(rr) >= 4 else np.nan,
        f'{p}rrShannon': rr_shannon_entropy(rr)         if len(rr) >= 4 else np.nan,
        f'{p}TPR':       turning_point_ratio(rr)        if len(rr) >= 3 else np.nan,
        f'{p}DFA_a1':    dfa_alpha1(rr)                 if len(rr) >= 4 else np.nan,
    }
    block.update(poincare_lorenz_features(rr, prefix=prefix))
    return block


# ── RR-Quellen zum Einstecken (liefern RR-Serie in ms oder None) ───────────

def _rr_ms_from_detector(signal, fs, peak_detector,
                         rr_min_ms=RR_MIN_MS, rr_max_ms=RR_MAX_MS, cv_max=None):
    """RR-Serie [ms] aus einem Peak-Detektor; None bei Fehlschlag.
    Gleiche Plausibilitätslogik wie hrv_from_detector → konsistente RR."""
    try:
        peaks = np.asarray(peak_detector(signal, fs))
    except Exception:
        return None
    if len(peaks) < 5:
        return None
    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr > rr_min_ms) & (rr < rr_max_ms)]
    if len(rr) < 4:
        return None
    if cv_max is not None and np.mean(rr) > 0 and np.std(rr) / np.mean(rr) > cv_max:
        return None
    return rr


def _rr_ms_heartpy(signal, fs):
    """RR-Serie [ms] über heartpy (gleiche Schläge wie hrv_heartpy). None bei Fehlschlag."""
    try:
        import heartpy as hp
    except ImportError:
        raise ImportError("heartpy nicht installiert: pip install heartpy")
    try:
        wd, m = hp.process(hp.scale_data(signal), sample_rate=fs,
                           bpmmin=30, bpmmax=220, reject_segmentwise=True)
        rr = np.asarray(wd.get('RR_list_cor', wd.get('RR_list', [])), dtype=float)
        rr = rr[(rr > RR_MIN_MS) & (rr < RR_MAX_MS)]
        return rr if len(rr) >= 4 else None
    except Exception:
        return None


def af_rr_from_detector(signal, fs, peak_detector, prefix='', cv_max=None):
    rr = _rr_ms_from_detector(signal, fs, peak_detector, cv_max=cv_max)
    return rr_af_feature_block(rr, prefix) if rr is not None else None


def af_rr_heartpy(signal, fs, prefix=''):
    rr = _rr_ms_heartpy(signal, fs)
    return rr_af_feature_block(rr, prefix) if rr is not None else None


def af_rr_bcg(signal, fs, prefix=''):
    return af_rr_from_detector(signal, fs, detect_peaks_bcg_cwt, prefix=prefix, cv_max=0.20)


def af_rr_cecg_cwt(signal, fs, prefix=''):
    return af_rr_from_detector(signal, fs, detect_peaks_cecg_cwt, prefix=prefix)


# ──────────────────────────────────────────────────────────────────────────
# 5. Bündelung pro Signal
# ──────────────────────────────────────────────────────────────────────────

def signal_feature_block(signal, fs, prefix, hrv_fn=None, af_rr_fn=None,
                         use_legacy_freq=False):
    """
    Erzeugt den kompletten Feature-Block EINES Signals (ein Fenster).

    hrv_fn   : callable(signal, fs, prefix) -> dict|None   (z.B. hrv_heartpy,
               hrv_bcg, hrv_cecg_cwt, oder None für peak-freie Signale)
    af_rr_fn : OPTIONAL callable(signal, fs, prefix) -> dict|None für die
               AF-RR-Features (z.B. af_rr_heartpy, af_rr_bcg, af_rr_cecg_cwt).
               Wenn None, werden KEINE AF-RR-Spalten angehängt (rückwärts-
               kompatibel zum bisherigen 20-Feature-Block).

    HRV- (und ggf. AF-RR-)Schlüssel sind immer vorhanden (NaN bei Fehlschlag),
    damit alle Fenster dieselbe Spaltenstruktur haben.
    """
    block = {}
    block.update(time_domain_features(signal, prefix))
    if use_legacy_freq:
        block.update(frequency_domain_features_legacy(signal, fs, prefix))
    else:
        block.update(frequency_domain_features(signal, fs, prefix))
    block.update(sample_entropy_signal(signal, prefix))

    hrv = hrv_fn(signal, fs, prefix=prefix) if hrv_fn is not None else None
    for k in HRV_KEYS:
        block[f'{prefix}_{k}'] = hrv[f'{prefix}_{k}'] if hrv else np.nan

    if af_rr_fn is not None:
        af = af_rr_fn(signal, fs, prefix=prefix)
        for k in AF_RR_KEYS:
            block[f'{prefix}_{k}'] = af[f'{prefix}_{k}'] if af else np.nan
    return block


# ──────────────────────────────────────────────────────────────────────────
# Selbsttest
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    fs = 128
    t  = np.arange(0, 30, 1 / fs)
    # Synthetisches „Puls"-Signal: 1.2 Hz (72 bpm) + Oberwelle + Rauschen
    rng = np.random.default_rng(0)
    sig = (np.sin(2 * np.pi * 1.2 * t)
           + 0.3 * np.sin(2 * np.pi * 2.4 * t)
           + 0.1 * rng.standard_normal(len(t)))

    print("── time_domain_features ──")
    for k, v in time_domain_features(sig, 'ppg1').items():
        print(f"  {k:22s} {v:+.4f}")

    print("\n── frequency_domain_features (neu) ──")
    for k, v in frequency_domain_features(sig, fs, 'ppg1').items():
        print(f"  {k:24s} {v}")

    print("\n── sample_entropy_signal ──")
    print("  ", sample_entropy_signal(sig, 'ppg1'))

    print("\n── HRV via austauschbarem Detektor (simple) ──")
    hrv = hrv_from_detector(sig, fs, detect_peaks_simple, prefix='ppg1')
    for k, v in (hrv or {}).items():
        print(f"  {k:14s} {v:.4f}")

    print("\n── HRV via CWT-cECG-Detektor (auf demselben Testsignal) ──")
    hrv2 = hrv_cecg_cwt(sig, fs, prefix='cecg')
    print("  ", {k: round(v, 4) for k, v in (hrv2 or {}).items()})

    print("\n── kompletter signal_feature_block (PPG, simple-HRV) ──")
    blk = signal_feature_block(
        sig, fs, 'ppg1',
        hrv_fn=lambda s, f, prefix: hrv_from_detector(s, f, detect_peaks_simple, prefix=prefix))
    print(f"  {len(blk)} Features: {sorted(blk.keys())}")

    print("\n── NEU: AF-RR-Features, Sinus vs. simuliertes AF ──")
    rng2 = np.random.default_rng(1)
    nrr = 45
    tt = np.arange(nrr)
    rr_sinus = 800 + 20 * np.sin(2 * np.pi * tt / 8) + rng2.normal(0, 8, nrr)
    rr_af = np.clip(800 + rng2.normal(0, 110, nrr), 350, 1500)
    bs = rr_af_feature_block(rr_sinus, 'x')
    ba = rr_af_feature_block(rr_af, 'x')
    print(f"  {'Feature':<14}{'Sinus':>10}{'AF':>10}   erwartet")
    richtung = {'x_CoSEn': 'AF↑', 'x_rrShannon': 'AF↑', 'x_TPR': 'AF→0.667',
                'x_DFA_a1': 'AF↓', 'x_Lorenz_occ': 'AF↑',
                'x_Lorenz_origin': 'AF↓', 'x_dRR_SD': 'AF↑'}
    for k in bs:
        print(f"  {k:<14}{bs[k]:>10.3f}{ba[k]:>10.3f}   {richtung[k]}")
    assert ba['x_CoSEn'] > bs['x_CoSEn'], "CoSEn-Richtung falsch"
    assert ba['x_Lorenz_occ'] > bs['x_Lorenz_occ'], "Lorenz-Richtung falsch"
    assert ba['x_DFA_a1'] < bs['x_DFA_a1'], "DFA-Richtung falsch"
    assert ba['x_rrShannon'] > bs['x_rrShannon'], "Shannon-Richtung falsch"

    print("\n── signal_feature_block MIT af_rr_fn (AF-RR-Spalten aktiv) ──")
    blk2 = signal_feature_block(
        sig, fs, 'ppg1',
        hrv_fn=lambda s, f, prefix: hrv_from_detector(s, f, detect_peaks_simple, prefix=prefix),
        af_rr_fn=lambda s, f, prefix: af_rr_from_detector(s, f, detect_peaks_simple, prefix=prefix))
    neue = sorted(set(blk2) - set(blk))
    print(f"  {len(blk2)} Features (vorher {len(blk)}); neue AF-RR-Spalten: {neue}")
    print("\n✓ Selbsttest erfolgreich – alle Funktionen laufen.")
