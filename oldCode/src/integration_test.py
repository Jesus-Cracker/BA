"""
integration_test.py — End-to-End-Test der gesamten Pipeline
===========================================================
Prüft, ob features.py, sqi.py und models.py korrekt ineinandergreifen.

Da hier keine echten Patientendaten vorliegen, werden REALISTISCHE synthetische
Multimodal-Signale erzeugt:
  - Non-AF: regelmäßige RR-Intervalle (niedrige HRV)
  - AF:     stark schwankende RR-Intervalle (hohe HRV) + gelegentliche Pausen
  - 5 Signale je Fenster (ppg1, ppg2, cecg, bcg1, bcg2), aus denselben Schlagzeiten
  - ARTEFAKT-Fenster: in ~35% der Fenster werden Signale verrauscht -> Peak-Detektion
    unzuverlässig, Features irreführend. Diese sollen vom SQI erkannt werden.

Erwartung: Die SQI-gewichtete Fusion ist robuster (>= ungewichtet), v.a. bei der
Spezifität, weil verrauschte Non-AF-Fenster sonst AF-ähnlich aussehen.

Aufruf:  python integration_test.py
"""

import numpy as np
import pandas as pd

import oldCode.src.features as F
import oldCode.src.sqi as Q
import oldCode.src.models as M

RNG = np.random.default_rng(7)
FS  = 128
DUR = 20.0          # s pro Fenster
SIGNALS = ['ppg1', 'ppg2', 'cecg', 'bcg1', 'bcg2']
SIG_TYPE = {'ppg1': 'ppg', 'ppg2': 'ppg', 'cecg': 'cecg', 'bcg1': 'bcg', 'bcg2': 'bcg'}


# ──────────────────────────────────────────────────────────────────────────
# Synthetische Signalerzeugung
# ──────────────────────────────────────────────────────────────────────────

def _template(fs, width=0.8, kind='ppg'):
    tt = np.linspace(-width / 2, width / 2, int(width * fs))
    if kind == 'cecg':                      # scharfer QRS-artiger Ausschlag
        return np.exp(-(tt / 0.02) ** 2)
    if kind == 'bcg':                        # breiter, biphasisch
        return np.sin(2 * np.pi * tt / width) * np.exp(-(tt / 0.3) ** 2)
    return (np.exp(-(tt / 0.05) ** 2)        # PPG: systolisch + dikrotisch
            + 0.3 * np.exp(-((tt - 0.15) / 0.10) ** 2))


def _beat_times(is_af, dur, rng):
    """Schlagzeiten: regelmäßig (Non-AF) oder irregulär (AF)."""
    mean_rr = rng.uniform(0.7, 0.95)         # ~63-85 bpm
    times, t = [], rng.uniform(0, 0.5)
    while t < dur:
        if is_af:
            rr = mean_rr * rng.uniform(0.6, 1.5)        # starke Variabilität
            if rng.random() < 0.10:
                rr *= 1.6                                # gelegentliche Pause
        else:
            rr = mean_rr + rng.normal(0, 0.02)          # nahezu konstant
        times.append(t); t += max(0.3, rr)
    return np.array(times)


def _make_signal(beat_times, fs, dur, kind):
    n = int(dur * fs)
    x = np.zeros(n)
    tmpl = _template(fs, kind=kind); h = len(tmpl) // 2
    for bt in beat_times:
        c = int(bt * fs); lo = max(0, c - h); hi = min(n, c - h + len(tmpl))
        if hi > lo:
            x[lo:hi] += tmpl[(lo - (c - h)):(lo - (c - h)) + (hi - lo)]
    return x + rng_noise(n, 0.05)


def rng_noise(n, sd):
    return RNG.normal(0, sd, n)


def _corrupt(sig):
    """Bewegungsartefakt: mehrere Rausch-Bursts."""
    out = sig.copy()
    for _ in range(RNG.integers(4, 9)):
        i = RNG.integers(0, len(out) - 60)
        out[i:i + 60] += RNG.normal(0, 8, 60)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Datensatz aufbauen + Pipeline durchlaufen
# ──────────────────────────────────────────────────────────────────────────

def build_dataset(n_patients=20, windows_per_patient=16):
    # HRV über injizierten einfachen Detektor (test-only, kein heartpy nötig)
    hrv_fn = lambda s, fs, prefix: F.hrv_from_detector(
        s, fs, F.detect_peaks_simple, prefix=prefix)

    rows, labels, groups, sqi_weights = [], [], [], []
    n_artifact = 0

    for pid in range(n_patients):
        is_af = pid % 2
        pat = f'PAT{pid:03d}'
        for w in range(windows_per_patient):
            artifact = RNG.random() < 0.35
            n_artifact += artifact
            bt = _beat_times(is_af, DUR, RNG)

            sigs = {}
            for s in SIGNALS:
                raw = _make_signal(bt, FS, DUR, SIG_TYPE[s])
                # Artefakt trifft 3 der 5 Signale
                if artifact and s in ('ppg1', 'cecg', 'bcg1'):
                    raw = _corrupt(raw)
                sigs[s] = raw

            # --- Features (features.py) ---
            feat = {}
            for s in SIGNALS:
                feat.update(F.signal_feature_block(sigs[s], FS, s, hrv_fn=hrv_fn))

            # --- SQI je Signal -> Fenstergewicht (sqi.py) ---
            comps = [Q.signal_sqi(sigs[s], FS, SIG_TYPE[s])['composite'] for s in SIGNALS]
            window_weight = float(np.mean(comps))

            rows.append(feat); labels.append(is_af); groups.append(pat)
            sqi_weights.append(window_weight)

    df = pd.DataFrame(rows)
    return df, np.array(labels), np.array(groups), np.array(sqi_weights), n_artifact


def main():
    print("1) Synthetischen Multimodal-Datensatz erzeugen…")
    df, y, groups, w_sqi, n_art = build_dataset()
    print(f"   Feature-Matrix: {df.shape[0]} Fenster × {df.shape[1]} Features")
    print(f"   Patienten: {len(np.unique(groups))}  |  AF-Fenster: {y.sum()}  Non-AF: {(1-y).sum()}")
    print(f"   Artefakt-Fenster: {n_art} ({n_art/len(y)*100:.0f}%)")
    print(f"   NaN-Anteil in Features: {df.isna().mean().mean()*100:.1f}%")

    # Plausibilitätscheck: erkennt der SQI die Artefaktfenster?
    # (nur für den Test verfügbar, da wir hier wissen wo Artefakte sind)
    print(f"\n2) SQI-Check: mittleres Fenstergewicht = {w_sqi.mean():.3f} "
          f"(min {w_sqi.min():.3f}, max {w_sqi.max():.3f})")

    print("\n3) Feature-Selektion (RF-Importance >= 0.008)…")
    mask, imp = M.select_features_rf(df.values, y, threshold=0.008)
    X_sel = df.values[:, mask]
    print(f"   {mask.sum()} von {df.shape[1]} Features selektiert")
    top = np.array(df.columns)[np.argsort(imp)[::-1][:8]]
    print(f"   Top-Features: {list(top)}")

    print("\n4) Modellvergleich — ungewichtet vs. SQI-gewichtet (nested, leckagefrei)…\n")
    res = M.compare_models(
        X_sel, y, groups, balanced=True,
        modes=('mv_nested', 'mv_nested_sqi'),
        window_sqi=w_sqi, min_sqi=0.0)

    fmt = {c: '{:.3f}'.format for c in
           ['Accuracy', 'AUC', 'Sensitivität', 'Spezifität', 'Threshold']}
    print(res.to_string(index=False, formatters=fmt))

    # Robustheitsgewinn zusammenfassen
    print("\n5) Robustheitsgewinn durch SQI-Gewichtung (Mittel über alle 4 Modelle):")
    for metric in ['Accuracy', 'AUC', 'Sensitivität', 'Spezifität']:
        plain = res[res.Modus == 'mv_nested'][metric].mean()
        sqiw  = res[res.Modus == 'mv_nested_sqi'][metric].mean()
        arrow = '↑' if sqiw > plain + 1e-9 else ('=' if abs(sqiw-plain) <= 1e-9 else '↓')
        print(f"   {metric:14s} ungewichtet={plain:.3f}  SQI-gewichtet={sqiw:.3f}  {arrow}")

    print("\n✓ End-to-End-Lauf erfolgreich — alle drei Module greifen ineinander.")


if __name__ == '__main__':
    main()
