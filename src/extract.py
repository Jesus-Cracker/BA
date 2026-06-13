"""
extract.py — Fenster-Feature- & SQI-Extraktion (Track B: SQI-gated Mixture of Experts)
======================================================================================
Bachelorarbeit: AF-Detektion in kontaktlosen Signalen · Nik Büttner · RWTH Aachen

Zweck dieser Datei
------------------
Diese Datei erzeugt EINE aufgeräumte Tabelle (eine Zeile pro Zeitfenster), aus der
sich anschließend ALLES Weitere schneiden lässt:

    *  drei Experten-Modelle (cECG / PPG / BCG)  -> je ein Merkmals-Block
    *  das Gating-Netz                           -> die SQI-Spalten

Damit ist die Merkmals-/SQI-Berechnung sauber von Training & Auswertung getrennt
(eigenes Notebook `01_features_sqi.ipynb`). Das Training (`02_experts_gating.ipynb`)
lädt nur noch die fertige Tabelle und rechnet NICHT erneut die teuren Features.

Spaltenkonvention der Ausgabetabelle
------------------------------------
    Metadaten (mit Doppelpunkt-freien, klaren Namen):
        patient        str   Patienten-ID (Gruppenschlüssel für LOPO-CV!)
        AF             int   Patienten-Label (0/1) — gilt für ALLE Fenster des Patienten
        win_idx        int   laufender Fenster-Index innerhalb des Patienten
        t_start_s      float Startzeit des Fensters in Sekunden
        n_valid_hrv    int   Anzahl Signale, die in diesem Fenster eine HRV liefern

    Experten-Merkmale (prefix = Signalname, exakt wie in features.signal_feature_block):
        cecg_*                      -> Experte 1 (cECG)
        ppg1_* , ppg2_*             -> Experte 2 (PPG)
        bcg1_* , bcg2_*             -> Experte 3 (BCG)

    Gating-Eingang (SQI je Signal, prefix `sqi_`):
        sqi_<signal>_kSQI / sSQI / pSQI / bSQI / tSQI / composite

WICHTIG (Leakage):
    Das Patienten-Label wird auf jedes Fenster verteilt. Die Fenster werden NICHT
    patientenweise zusammengefasst — die Vorhersage erfolgt fensterweise. Der
    Train/Test-Split MUSS aber weiterhin patientenweise erfolgen (groups=patient,
    LeaveOneGroupOut), sonst landen Fenster desselben Patienten in Training UND
    Test -> Leakage. Diese Datei liefert dafür die `patient`-Spalte.

Diese Datei berechnet NUR. Sie trifft keine Modell-Entscheidungen.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Wiederverwendete Bausteine aus dem bestehenden, validierten Code
# (features.py / sqi.py / signal_loader.py bleiben UNVERÄNDERT)
# --------------------------------------------------------------------------
import features as F
import sqi as Q


# ──────────────────────────────────────────────────────────────────────────
# Signal- und Modalitäts-Definition
# ──────────────────────────────────────────────────────────────────────────

SIGNALS  = ['cecg', 'ppg1', 'ppg2', 'bcg1', 'bcg2']

# Bandwahl für den SQI (psQI/HR) je physikalischem Signaltyp
SIG_TYPE = {'cecg': 'cecg', 'ppg1': 'ppg', 'ppg2': 'ppg', 'bcg1': 'bcg', 'bcg2': 'bcg'}

# Gruppierung der fünf Signale auf die DREI Experten (Vorgabe Betreuer)
MODALITIES = {
    'cecg': ['cecg'],          # Experte 1
    'ppg':  ['ppg1', 'ppg2'],  # Experte 2
    'bcg':  ['bcg1', 'bcg2'],  # Experte 3
}

# Welche SQIs pro Signal gespeichert werden (alle GT-frei -> als Gate-Eingang nutzbar)
SQI_KEYS = ['kSQI', 'sSQI', 'pSQI', 'bSQI', 'tSQI', 'composite']

# Feature-Version: geht in den Cache-Hash ein. Bei JEDER Änderung an der
# Spaltenstruktur (neue Features) erhöhen -> alte Caches werden NICHT mehr
# stillschweigend wiederverwendet (verhindert "fehlende Spalten"-Fehler).
FEAT_VERSION = 'v5_xmodal_xcorr'  # 'v4_bcg_swt_papr'

# Standard-Detektoren je Signal (entsprechen dem bisher besten Stand).
# Es werden NAMEN (Strings) gespeichert, damit joblib/loky-Worker sie sicher
# über `getattr(F, name)` auflösen können (Funktionsobjekte sind heikel zu picklen).
HRV_FN_DEFAULT = {
    'cecg': 'hrv_cecg_cwt',     # CWT-Morlet (cECG); Alt.: 'hrv_heartpy'
    'ppg1': 'hrv_heartpy',
    'ppg2': 'hrv_heartpy',
    'bcg1': 'hrv_bcg_nogate',
    'bcg2': 'hrv_bcg_nogate',
}
AF_RR_FN_DEFAULT = {
    'cecg': 'af_rr_cecg_cwt',
    'ppg1': 'af_rr_heartpy',
    'ppg2': 'af_rr_heartpy',
    'bcg1': 'af_rr_bcg_nogate',
    'bcg2': 'af_rr_bcg_nogate',
}


# ──────────────────────────────────────────────────────────────────────────
# Konfiguration
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractConfig:
    """Alle Stellschrauben der Extraktion an einem Ort (reproduzierbar & cachebar)."""
    data_root: str                       # Ordner mit PAT*-Unterordnern
    af_list:   str                       # Textdatei: eine AF-Patienten-ID je Zeile

    fs:        int = 128                 # Abtastrate der kontaktlosen Signale [Hz]
    window_s:  int = 30                  # Fensterlänge [s]
    hop_s:     int = 15                  # Schrittweite [s]  (50 % Überlappung)

    hrv_fn:    dict = field(default_factory=lambda: dict(HRV_FN_DEFAULT))
    af_rr_fn:  dict = field(default_factory=lambda: dict(AF_RR_FN_DEFAULT))
    use_af_rr: bool = True               # AF-spezifische RR-Merkmale anhängen
    use_legacy_freq: bool = False        # alte LF/HF-Frequenzbänder (nur Reproduktion)

    # BCG-Merkmalspfad:
    #   'wavelet' (Standard): SWT+PAPR-Merkmale nach Yu et al. 2019 (KEINE RR/HRV).
    #                         Begründung s. features.bcg_wavelet_feature_block:
    #                         AF zeigt sich im BCG morphologisch (Clutter), nicht
    #                         über die unzuverlässige J-Peak-RR-Streuung.
    #   'rr'      (Altpfad) : bisherige HRV/af_rr-Merkmale aus J-Peak-Detektion
    #                         (für A/B-Vergleich in der Arbeit beibehalten).
    bcg_mode:  str = 'wavelet'

    # Fenster behalten, wenn >= min_valid_hrv Signale eine HRV liefern.
    # 0 = wirklich alle Fenster behalten (empfohlen für MoE: ein totes cECG-Fenster
    #     kann über ein gutes PPG-Fenster trotzdem korrekt klassifiziert werden).
    min_valid_hrv: int = 1

    @property
    def win(self) -> int:
        return self.window_s * self.fs

    @property
    def hop(self) -> int:
        return self.hop_s * self.fs

    def cache_name(self, results_dir: str) -> str:
        """Eindeutiger Cache-Dateiname aus der Feature-Konfiguration."""
        import hashlib
        cfg = '|'.join(
            [self.hrv_fn[s] for s in SIGNALS]
            + [self.af_rr_fn[s] for s in SIGNALS]
            + [str(self.use_af_rr), str(self.use_legacy_freq), str(self.bcg_mode),
               str(self.window_s), str(self.hop_s), str(self.min_valid_hrv),
               FEAT_VERSION]
        )
        tag = self.hrv_fn['cecg'].replace('hrv_', '')
        h   = hashlib.sha1(cfg.encode()).hexdigest()[:6]
        return os.path.join(results_dir, f'features_sqi_{tag}_{h}.csv')

    def as_dict(self) -> dict:
        """Picklebare Repräsentation für joblib-Worker."""
        return dict(
            data_root=self.data_root, fs=self.fs, win=self.win, hop=self.hop,
            hrv_fn=self.hrv_fn, af_rr_fn=self.af_rr_fn,
            use_af_rr=self.use_af_rr, use_legacy_freq=self.use_legacy_freq,
            bcg_mode=self.bcg_mode,
            min_valid_hrv=self.min_valid_hrv,
        )


# ──────────────────────────────────────────────────────────────────────────
# Kern: EIN Fenster -> EINE Zeile (testbar ohne Dateizugriff)
# ──────────────────────────────────────────────────────────────────────────

def extract_window(sig_windows: dict, fs: int,
                   hrv_fn: dict, af_rr_fn: dict,
                   use_af_rr: bool = True, use_legacy_freq: bool = False,
                   bcg_mode: str = 'wavelet'):
    """
    Berechnet Merkmals-Block + SQIs für EIN Fenster.

    sig_windows : {signalname: 1D-np.ndarray}   (gefiltertes Fenster je Signal)
    hrv_fn      : {signalname: callable(signal, fs, prefix)->dict|None}
    af_rr_fn    : {signalname: callable(...)->dict|None}
    bcg_mode    : 'wavelet' -> BCG nutzt SWT+PAPR-Merkmale (Yu et al. 2019, KEINE
                  RR/HRV); 'rr' -> bisheriger HRV/af_rr-Pfad (A/B-Vergleich).

    Rückgabe: (feat: dict, n_valid_hrv: int)
        feat enthält die Experten-Merkmale ({signal}_*) UND die Gate-SQIs
        (sqi_{signal}_*). KEINE Metadaten (die hängt der Patienten-Loop an).
    """
    feat = {}
    n_valid = 0
    hr_per_signal = {}
    for s in SIGNALS:
        w = sig_windows[s]

        # --- Experten-Merkmale (identische Spaltenstruktur über alle Fenster) ---
        if s in ('bcg1', 'bcg2') and bcg_mode == 'wavelet':
            # Paper-treuer BCG-Pfad: SWT+PAPR-Morphologie statt RR/HRV.
            block = F.bcg_wavelet_feature_block(w, fs, s)
        else:
            block = F.signal_feature_block(
                w, fs, s,
                hrv_fn=hrv_fn[s],
                af_rr_fn=af_rr_fn[s] if use_af_rr else None,
                use_legacy_freq=use_legacy_freq,
            )
        feat.update(block)

        # --- SQIs (Gate-Eingang) ---
        sq = Q.signal_sqi(w, fs, SIG_TYPE[s])
        for k in SQI_KEYS:
            feat[f'sqi_{s}_{k}'] = sq[k]

        # GT-freie HR-Schätzung je Signal (für den multimodalen Konsens unten)
        hr_per_signal[s] = Q.estimate_hr_fft(w, fs, band=Q.SQI_BANDS[SIG_TYPE[s]])

        # HRV-Gültigkeit (für n_valid_hrv / optionales Verwerfen)
        if np.isfinite(feat.get(f'{s}_meanRR', np.nan)):
            n_valid += 1

    # --- Multimodale HR-Selbstkonsistenz (GT-FREI -> zusätzlicher Gate-Eingang) ---
    # Einigen sich >=2 kontaktlose Signale auf dieselbe HR, ist das Fenster
    # vertrauenswürdig — ohne Goldstandard. Bisher in sqi.py vorhanden, aber NICHT
    # in der Tabelle. Spalten 'sqi_xmodal_*' werden von gate_sqi_cols('all') als
    # Gate-Eingang aufgenommen (enden nicht auf '_composite').
    xm = Q.cross_modal_hr_agreement(hr_per_signal, tol_bpm=10.0, min_agree=2)
    feat['sqi_xmodal_n_agree']     = float(xm['n_agree'])
    feat['sqi_xmodal_confidence']  = float(xm['confidence'])
    feat['sqi_xmodal_trustworthy'] = float(bool(xm['trustworthy']))

    # --- Inter-Kanal-Korrelation je Modalität (QUALITÄTS-Feature, AF-orthogonal) ---
    # ppg1/ppg2 bzw. bcg1/bcg2 sehen dasselbe physiologische Signal. Hohe Korrelation
    # = beide Kanäle erfassen konsistent dieselbe Wellenform (gute Qualität); sie
    # bleibt auch bei AF hoch (gleicher, nur unregelmäßiger Puls in beiden Kanälen)
    # -> erfüllt den "Phantom-Test" (perfektes SNR + irreguläre Schläge => weiterhin
    # hoch) und ist damit ein sauberer GATE-Eingang, kein AF-Leck.
    for mod, (a, b) in {'ppg': ('ppg1', 'ppg2'), 'bcg': ('bcg1', 'bcg2')}.items():
        feat[f'sqi_{mod}_xcorr'] = _safe_xcorr(sig_windows[a], sig_windows[b])

    return feat, n_valid


def _safe_xcorr(x, y):
    """Pearson-Korrelation zweier Kanäle, robust (NaN/Konstanz -> 0.0)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


# ──────────────────────────────────────────────────────────────────────────
# Ein Patient -> Liste von Zeilen  (läuft im joblib-Worker)
# ──────────────────────────────────────────────────────────────────────────

def _extract_one_patient(pid: str, cfgd: dict, af_set: set):
    """Worker-Funktion: lädt einen Patienten, filtert, fenstert, extrahiert."""
    # Worker importieren ihre eigenen Module — Pfad robust setzen
    for p in ['src', '.', '../src']:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'features.py')):
            if p not in sys.path:
                sys.path.insert(0, p)
            break
    import features as _F  # noqa: F401  (sicherstellen, dass importierbar)
    import sqi as _Q       # noqa: F401
    from signal_loader import PatientSignals

    hrv  = {s: getattr(F, cfgd['hrv_fn'][s]) for s in SIGNALS}
    afrr = {s: getattr(F, cfgd['af_rr_fn'][s]) for s in SIGNALS}

    try:
        pat = PatientSignals(os.path.join(cfgd['data_root'], pid))
        pat.filter_all(fs=cfgd['fs'])
        pat.offset_correction()
    except Exception as e:
        return pid, [], f'Ladefehler: {e}'

    is_af = int(pid in af_set)
    win, hop, fs = cfgd['win'], cfgd['hop'], cfgd['fs']
    n_fen = (len(pat.cecg_filt) - win) // hop + 1

    rows = []
    for i in range(n_fen):
        start = i * hop
        sig_windows = {s: getattr(pat, f'{s}_filt')[start:start + win] for s in SIGNALS}
        feat, n_valid = extract_window(
            sig_windows, fs, hrv, afrr,
            use_af_rr=cfgd['use_af_rr'], use_legacy_freq=cfgd['use_legacy_freq'],
            bcg_mode=cfgd.get('bcg_mode', 'wavelet'))

        if n_valid < cfgd['min_valid_hrv']:
            continue

        feat.update({
            'patient': pid, 'AF': is_af, 'win_idx': i,
            't_start_s': start / fs, 'n_valid_hrv': n_valid,
        })
        rows.append(feat)

    return pid, rows, None


# ──────────────────────────────────────────────────────────────────────────
# Datensatz-Extraktion (parallel) + Caching
# ──────────────────────────────────────────────────────────────────────────

def extract_dataset(cfg: ExtractConfig, n_jobs: int = 8, verbose: bool = True) -> pd.DataFrame:
    """
    Extrahiert ALLE Patienten parallel und gibt die aufgeräumte Tabelle zurück.
    Nutzt loky-Backend mit inner_max_num_threads=1 (kein Thread-Oversubscribing).
    """
    from joblib import Parallel, delayed, parallel_config

    with open(cfg.af_list) as f:
        af_set = {l.strip() for l in f if l.strip()}

    patients = sorted(d for d in os.listdir(cfg.data_root)
                      if os.path.isdir(os.path.join(cfg.data_root, d)) and d.startswith('PAT'))

    # Schutz: Worker importieren features.py — Detektoren müssen DORT existieren
    for s in SIGNALS:
        if not hasattr(F, cfg.hrv_fn[s]):
            raise RuntimeError(f"HRV-Funktion '{cfg.hrv_fn[s]}' fehlt in features.py")
        if cfg.use_af_rr and not hasattr(F, cfg.af_rr_fn[s]):
            raise RuntimeError(f"AF-RR-Funktion '{cfg.af_rr_fn[s]}' fehlt in features.py")
    if cfg.bcg_mode == 'wavelet' and not hasattr(F, 'bcg_wavelet_feature_block'):
        raise RuntimeError("bcg_wavelet_feature_block fehlt in features.py")

    t0 = time.time()
    cfgd = cfg.as_dict()
    with parallel_config(backend='loky', n_jobs=n_jobs, inner_max_num_threads=1):
        out = Parallel()(delayed(_extract_one_patient)(pid, cfgd, af_set) for pid in patients)

    rows = []
    for pid, prows, err in out:
        if err:
            if verbose:
                print(f'  {pid}: {err}')
            continue
        if verbose:
            print(f'  {pid} ({"AF" if pid in af_set else "Non-AF"}): {len(prows)} Fenster')
        rows.extend(prows)

    df = pd.DataFrame(rows)
    # Metadaten nach vorne sortieren — reine Kosmetik, erleichtert das Lesen
    meta = ['patient', 'AF', 'win_idx', 't_start_s', 'n_valid_hrv']
    df = df[[c for c in meta if c in df.columns]
            + [c for c in df.columns if c not in meta]]
    if verbose:
        print(f'  gesamt {len(df)} Fenster · {df.shape[1]} Spalten · {time.time()-t0:.1f}s')
    return df


def load_or_extract(cfg: ExtractConfig, results_dir: str, n_jobs: int = 8,
                    force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Lädt den Cache, falls vorhanden; extrahiert sonst und speichert ihn."""
    os.makedirs(results_dir, exist_ok=True)
    path = cfg.cache_name(results_dir)
    if os.path.exists(path) and not force:
        if verbose:
            print(f'Cache gefunden -> lade {path}')
        return pd.read_csv(path)
    if verbose:
        print('Kein Cache -> extrahiere ...')
    df = extract_dataset(cfg, n_jobs=n_jobs, verbose=verbose)
    df.to_csv(path, index=False)
    if verbose:
        print(f'Gespeichert: {path}')
    return df


# ──────────────────────────────────────────────────────────────────────────
# Spalten-Helfer: schneiden die Tabelle für Experten / Gate / Metadaten
# ──────────────────────────────────────────────────────────────────────────

META_COLS = ['patient', 'AF', 'win_idx', 't_start_s', 'n_valid_hrv']


def expert_feature_cols(df: pd.DataFrame, modality: str) -> list:
    """
    Merkmalsspalten EINES Experten.
        modality in {'cecg','ppg','bcg'}
    cECG -> alle 'cecg_*'; PPG -> 'ppg1_*'+'ppg2_*'; BCG -> 'bcg1_*'+'bcg2_*'
    SQI-Spalten ('sqi_*') sind bewusst AUSGESCHLOSSEN — die gehören dem Gate.
    """
    if modality not in MODALITIES:
        raise ValueError(f"Unbekannte Modalität '{modality}'. Erlaubt: {list(MODALITIES)}")
    sigs = MODALITIES[modality]
    cols = [c for c in df.columns
            if any(c.startswith(s + '_') for s in sigs) and not c.startswith('sqi_')]
    return cols


def gate_sqi_cols(df: pd.DataFrame, kind: str = 'all') -> list:
    """
    SQI-Spalten für das Gating-Netz.
        kind='all'        -> alle 5 SQIs je Signal (5 Signale x 5 = 25 Eingänge)
        kind='composite'  -> nur der zusammengefasste composite-SQI je Signal (5 Eingänge)
    """
    if kind == 'composite':
        return [c for c in df.columns if c.startswith('sqi_') and c.endswith('_composite')]
    if kind == 'all':
        return [c for c in df.columns
                if c.startswith('sqi_') and not c.endswith('_composite')]
    raise ValueError("kind muss 'all' oder 'composite' sein")


def add_neighbor_context(df: pd.DataFrame, cols: list | None = None, k: int = 1) -> pd.DataFrame:
    """Hängt je SQI-Spalte den Wert des VORIGEN und NÄCHSTEN Fensters DESSELBEN
    Patienten an (zeitlicher Kontext -> Gewichts-Glättung). Neue Spalten heißen
    '<col>_prev'/'<col>_next' und beginnen mit 'sqi_' -> werden automatisch von
    gate_sqi_cols('all') als zusätzliche Gate-Eingänge aufgenommen.

    Leckagefrei: der Shift erfolgt INNERHALB eines Patienten (groupby 'patient'),
    am Rand wird mit dem eigenen Fensterwert aufgefüllt (kein Fremdpatient).
    cols=None -> alle nicht-composite SQI-Spalten (die aktuellen Gate-Eingänge).
    k -> Versatz in Fenstern (1 = direkte Nachbarn).
    """
    out = df.copy()
    if cols is None:
        cols = gate_sqi_cols(df, 'all')
    cols = [c for c in cols if c in out.columns and c.startswith('sqi_')]
    if 'win_idx' in out.columns:
        out = out.sort_values(['patient', 'win_idx'])
    g = out.groupby('patient', sort=False)
    for c in cols:
        out[f'{c}_prev'] = g[c].shift(k).fillna(out[c])
        out[f'{c}_next'] = g[c].shift(-k).fillna(out[c])
    return out.loc[df.index]


def split_Xygroups(df: pd.DataFrame):
    """Bequemer Zugriff auf das, was die CV-Routinen brauchen."""
    y      = df['AF'].astype(int).values
    groups = df['patient'].values
    return df, y, groups


# ──────────────────────────────────────────────────────────────────────────
# Selbsttest (ohne Daten / ohne signal_loader) — prüft Spaltenstruktur
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    fs = 128
    t = np.arange(0, 30, 1 / fs)
    rng = np.random.default_rng(0)
    base = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)

    sig_windows = {s: base + 0.05 * rng.standard_normal(t.size) for s in SIGNALS}
    hrv  = {s: getattr(F, HRV_FN_DEFAULT[s]) for s in SIGNALS}
    afrr = {s: getattr(F, AF_RR_FN_DEFAULT[s]) for s in SIGNALS}

    feat, nval = extract_window(sig_windows, fs, hrv, afrr)
    df = pd.DataFrame([{**feat, 'patient': 'PAT000', 'AF': 1,
                        'win_idx': 0, 't_start_s': 0.0, 'n_valid_hrv': nval}])

    print('Spalten gesamt:', df.shape[1])
    print('n_valid_hrv   :', nval)
    for m in MODALITIES:
        print(f'  Experte {m:5s}: {len(expert_feature_cols(df, m))} Merkmale')
    print('  Gate (all)   :', len(gate_sqi_cols(df, "all")), 'SQI-Eingänge')
    print('  Gate (comp.) :', len(gate_sqi_cols(df, "composite")), 'SQI-Eingänge')
    print('Selbsttest OK.')
