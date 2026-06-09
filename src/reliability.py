"""
reliability.py — GT-ECG-basiertes Zuverlässigkeits-Ziel je Fenster & Modalität
==============================================================================
Bachelorarbeit: AF-Detektion in kontaktlosen Signalen · Nik Büttner · RWTH Aachen
Track B: SQI-gated Mixture of Experts (Bachelet-Stil)

Was dieses Modul liefert
------------------------
Für jedes Fenster und jede Modalität (cECG / PPG / BCG) einen **Fehlerwert**, der
beschreibt, wie treu die Modalität den WAHREN Rhythmus (GT-EKG) abbildet. Genau
dieses Ziel sagt das Gating-Netz später aus den SQIs voraus (Bachelet:
"Fehlerprädiktion mittels eines KNN").

Wichtige Abgrenzung — RR-IRREGULARITÄT statt nur HR
---------------------------------------------------
Bachelet schätzte die Herzfrequenz; sein Fehler war |HR − HR_GT|. AF ist aber eine
RHYTHMUS-Frage: eine Modalität kann die mittlere HR exakt treffen und die
Unregelmäßigkeit trotzdem völlig verfehlen — und damit für die AF-Erkennung
nutzlos sein. Das Default-Ziel ist deshalb der Fehler in der **CoSEn**
(Coefficient of Sample Entropy, dem stärksten AF-Diskriminator deiner Features),
NICHT der HR-Fehler. Der HR-Fehler wird zur Vergleichbarkeit mit Bachelet
mitgeführt, ist aber nicht das Standard-Ziel.

GT-frei? Nein — und das ist genau der Punkt
-------------------------------------------
Das GT-EKG existiert NUR in Studie/Training, nicht im Einsatz (kontaktlos = keine
Elektrode). Es darf das Gate also TRAINIEREN, aber niemals dessen Eingang sein.
Diese Datei berechnet das Trainings-ZIEL aus dem GT-EKG; der Gate-EINGANG bleibt
der GT-freie SQI (siehe extract.gate_sqi_cols).

Spalten der Ausgabetabelle (join über patient + win_idx an die Feature-Tabelle):
    rel_<mod>_valid      bool   Modalität hat eine plausible RR-Serie geliefert
    rel_<mod>_hr_err     float  |HR_mod − HR_GT|  [bpm]      (Bachelet-Vergleich)
    rel_<mod>_cosen_err  float  |CoSEn_mod − CoSEn_GT|       (AF-Default-Ziel)
    rel_<mod>_drr_sd_err float  |dRR_SD_mod − dRR_SD_GT| [ms] (AF-Alternative)
    rel_<mod>_target     float  gewähltes Ziel (Default = cosen_err)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

import features as F


MODALITIES = {'cecg': ['cecg'], 'ppg': ['ppg1', 'ppg2'], 'bcg': ['bcg1', 'bcg2']}
ERROR_METRICS = ['hr_err', 'cosen_err', 'drr_sd_err']


# ──────────────────────────────────────────────────────────────────────────
# RR-Serien
# ──────────────────────────────────────────────────────────────────────────

def _rpeaks_fallback(w_gt, fs_gt):
    """Robuste R-Zacken-Detektion ohne neurokit2 (für Portabilität/Tests).
    Bandpass 5–15 Hz → quadrieren → find_peaks mit physiologischem Mindestabstand."""
    w = np.asarray(w_gt, dtype=float)
    b, a = butter(2, [5, 15], fs=fs_gt, btype='bandpass')
    sq = filtfilt(b, a, w) ** 2
    thr = np.mean(sq) + 0.5 * np.std(sq)
    peaks, _ = find_peaks(sq, height=thr, distance=int(0.3 * fs_gt))  # max ~200 bpm
    return peaks


def gt_rr_ms(w_gt, fs_gt: int = 500):
    """
    GT-RR-Serie [ms] aus dem GT-EKG-Fenster.
    Bevorzugt neurokit2 (wie in der bisherigen Validierung, Zelle 10), sonst
    Fallback-Detektor. None, wenn zu wenige plausible RR.
    """
    try:
        import neurokit2 as nk
        _, info = nk.ecg_process(np.asarray(w_gt, dtype=float), sampling_rate=fs_gt)
        rpk = np.asarray(info['ECG_R_Peaks'])
    except Exception:
        rpk = _rpeaks_fallback(w_gt, fs_gt)
    if len(rpk) < 5:
        return None
    rr = np.diff(rpk) / fs_gt * 1000.0
    rr = rr[(rr > F.RR_MIN_MS) & (rr < F.RR_MAX_MS)]
    return rr if len(rr) >= 4 else None


def modality_rr_ms(signal, fs, modality: str):
    """
    RR-Serie [ms] einer Modalität — mit DENSELBEN Detektoren wie die Features,
    damit das Ziel exakt zur Sichtweise des Experten passt.
    """
    if modality == 'cecg':
        return F._rr_ms_from_detector(signal, fs, F.detect_peaks_cecg_cwt)
    if modality == 'ppg':
        return F._rr_ms_heartpy(signal, fs)
    if modality == 'bcg':
        return F._rr_ms_from_detector(signal, fs, F.detect_peaks_bcg_cwt, cv_max=0.20)
    raise ValueError(f"Unbekannte Modalität '{modality}'")


# ──────────────────────────────────────────────────────────────────────────
# Fehler-Metriken (reine Funktion — direkt testbar)
# ──────────────────────────────────────────────────────────────────────────

def _hr_bpm(rr):
    return 60000.0 / np.median(rr) if (rr is not None and len(rr) >= 4) else np.nan


def _cosen(rr):
    return F.coefficient_sample_entropy(np.asarray(rr, float)) if (rr is not None and len(rr) >= 4) else np.nan


def _drr_sd(rr):
    return float(np.std(np.diff(np.asarray(rr, float)))) if (rr is not None and len(rr) >= 4) else np.nan


def rr_error_metrics(rr_mod, rr_gt) -> dict:
    """
    Fehler zwischen Modalitäts-RR und GT-RR. NaN, wenn eine Seite fehlt.
    valid = beide Seiten lieferten eine plausible RR-Serie.
    """
    valid = (rr_mod is not None and rr_gt is not None
             and len(rr_mod) >= 4 and len(rr_gt) >= 4)

    def _absdiff(f):
        a, b = f(rr_mod), f(rr_gt)
        return abs(a - b) if (np.isfinite(a) and np.isfinite(b)) else np.nan

    return {
        'valid':       bool(valid),
        'hr_err':      _absdiff(_hr_bpm),
        'cosen_err':   _absdiff(_cosen),
        'drr_sd_err':  _absdiff(_drr_sd),
    }


# ──────────────────────────────────────────────────────────────────────────
# Ein Fenster -> Zuverlässigkeits-Ziel je Modalität
# ──────────────────────────────────────────────────────────────────────────

def window_reliability(modality_signals: dict, w_gt, fs: int, fs_gt: int = 500,
                        target_metric: str = 'cosen_err') -> dict:
    """
    modality_signals : {'cecg': arr, 'ppg': [arr1, arr2], 'bcg': [arr1, arr2]}
                       (bereits gefilterte Fenster, 128 Hz)
    w_gt             : GT-EKG-Fenster (gefiltert, 500 Hz), zeitlich deckungsgleich
    target_metric    : welche Metrik das 'rel_<mod>_target' füllt (Default cosen_err)

    Aggregation Mehrkanal (PPG/BCG): die Modalität ist so zuverlässig wie ihr
    BESTER Kanal — der Experte kann den jeweils besseren Kanal ausnutzen.
    Daher: kleinster Fehler über die Kanäle (bei Gleichstand der valide Kanal).
    """
    if target_metric not in ERROR_METRICS:
        raise ValueError(f"target_metric muss aus {ERROR_METRICS} sein")
    rr_gt = gt_rr_ms(w_gt, fs_gt)

    out = {}
    for m, sigs in modality_signals.items():
        sigs = sigs if isinstance(sigs, (list, tuple)) else [sigs]
        cand = [rr_error_metrics(modality_rr_ms(s, fs, m), rr_gt) for s in sigs]

        # bester Kanal nach Zielmetrik (NaN ans Ende), sonst erster valider
        def _key(c):
            v = c[target_metric]
            return (np.isnan(v), v if np.isfinite(v) else np.inf)
        best = sorted(cand, key=_key)[0]

        out[f'rel_{m}_valid']      = any(c['valid'] for c in cand)
        out[f'rel_{m}_hr_err']     = best['hr_err']
        out[f'rel_{m}_cosen_err']  = best['cosen_err']
        out[f'rel_{m}_drr_sd_err'] = best['drr_sd_err']
        out[f'rel_{m}_target']     = best[target_metric]
    return out


# ──────────────────────────────────────────────────────────────────────────
# Datensatz-weite Reliability-Tabelle (parallel) — Gegenstück zu extract.py
# ──────────────────────────────────────────────────────────────────────────

def _reliability_one_patient(pid, cfgd, target_metric):
    """Worker: lädt einen Patienten, fenstert, berechnet je Fenster das Gate-Ziel.
    Verwendet DIESELBE Fensterung wie extract.py (gleiche win_idx -> join möglich)."""
    import os, sys
    for p in ['src', '.', '../src']:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'features.py')):
            if p not in sys.path:
                sys.path.insert(0, p)
            break
    from signal_loader import PatientSignals

    fs, fs_gt = cfgd['fs'], cfgd['fs_gt']
    win, hop = cfgd['win'], cfgd['hop']
    win_gt = cfgd['window_s'] * fs_gt
    try:
        pat = PatientSignals(os.path.join(cfgd['data_root'], pid))
        pat.filter_all(fs=fs)
        pat.offset_correction()
    except Exception as e:
        return pid, [], f'Ladefehler: {e}'
    if pat.gt_ecg_filt is None:
        return pid, [], 'kein GT-EKG'

    n_fen = (len(pat.cecg_filt) - win) // hop + 1
    rows = []
    for i in range(n_fen):
        start = i * hop
        gt0 = start * fs_gt // fs                      # zeitgleiches GT-Fenster
        w_gt = pat.gt_ecg_filt[gt0: gt0 + win_gt]
        if len(w_gt) < win_gt:
            break
        mod_sigs = {m: [getattr(pat, f'{s}_filt')[start:start + win] for s in sigs]
                    for m, sigs in MODALITIES.items()}
        rel = window_reliability(mod_sigs, w_gt, fs, fs_gt, target_metric=target_metric)
        rel.update({'patient': pid, 'win_idx': i})
        rows.append(rel)
    return pid, rows, None


def build_reliability_table(data_root, fs=128, fs_gt=500, window_s=30, hop_s=15,
                            target_metric='cosen_err', n_jobs=8, verbose=True):
    """
    Baut die Reliability-Tabelle über alle PAT*-Patienten (parallel).
    Join an die Feature-Tabelle später über ['patient', 'win_idx'].
    Benötigt neurokit2 für die GT-R-Zacken (sonst Fallback-Detektor).
    """
    import os, time
    from joblib import Parallel, delayed, parallel_config
    
    patients = sorted(d for d in os.listdir(data_root)
                      if os.path.isdir(os.path.join(data_root, d)) and d.startswith('PAT'))
    cfgd = dict(data_root=data_root, fs=fs, fs_gt=fs_gt, window_s=window_s,
                win=window_s * fs, hop=hop_s * fs)

    t0 = time.time()
    with parallel_config(backend='loky', n_jobs=n_jobs, inner_max_num_threads=1):
        out = Parallel()(delayed(_reliability_one_patient)(pid, cfgd, target_metric)
                         for pid in patients)
    rows = []
    for pid, prows, err in out:
        if err:
            if verbose:
                print(f'  {pid}: {err}')
            continue
        if verbose:
            print(f'  {pid}: {len(prows)} Fenster')
        rows.extend(prows)
    df = pd.DataFrame(rows)
    if verbose and len(df):
        cov = {m: f"{df[f'rel_{m}_valid'].mean() * 100:.0f}%" for m in MODALITIES}
        print(f'  gesamt {len(df)} Fenster · gültig je Modalität: {cov} · {time.time()-t0:.1f}s')
    return df


# ──────────────────────────────────────────────────────────────────────────
# Selbsttest: Metrik-Logik + GT-Detektion auf synthetischen Daten
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rng = np.random.default_rng(0)

    # 1) Reine Metrik-Logik mit BEKANNTEN RR-Serien
    rr_regular   = np.full(30, 800.0) + rng.normal(0, 5, 30)    # ~75 bpm, regelmäßig
    rr_irregular = rng.uniform(500, 1100, 30)                   # AF-artig, unregelmäßig

    m_good = rr_error_metrics(rr_regular + rng.normal(0, 8, 30), rr_regular)
    m_bad  = rr_error_metrics(rr_irregular, rr_regular)         # falscher Rhythmus
    m_none = rr_error_metrics(None, rr_regular)                 # Detektor versagt

    print('treuer Kanal  :', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m_good.items()})
    print('falscher Rhyth:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m_bad.items()})
    print('kein RR       :', m_none)
    assert m_good['cosen_err'] < m_bad['cosen_err'], 'treuer Kanal muss kleineren CoSEn-Fehler haben'
    assert m_none['valid'] is False

    # 2) GT-Detektion (Fallback) auf synthetischem R-Zacken-Zug @500 Hz
    fs_gt = 500
    rr_true = np.full(40, 800.0)                                # 75 bpm
    beat_t  = np.cumsum(rr_true) / 1000.0
    t = np.arange(0, beat_t[-1] + 1, 1 / fs_gt)
    ecg = np.zeros_like(t)
    for bt in beat_t:                                           # schmale R-Spikes
        ecg += np.exp(-0.5 * ((t - bt) / 0.012) ** 2)
    rr_gt = gt_rr_ms(ecg, fs_gt)
    print('\nGT-Fallback: %d RR, Median %.0f ms (~%.0f bpm)'
          % (len(rr_gt), np.median(rr_gt), 60000 / np.median(rr_gt)))
    assert rr_gt is not None and abs(np.median(rr_gt) - 800) < 30

    print('Selbsttest OK.')
