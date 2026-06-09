"""
sqi_validation.py — SQI-Trennschärfe, GT-Validierung & Kalibrierung
====================================================================
Bachelorarbeit: AF-Detektion in kontaktlosen Signalen · Nik Büttner · RWTH Aachen

Adressiert Problem 2: "Signale stark, aber GLEICHMÄSSIG verrauscht — verwirft
der SQI dann pauschal alles, obwohl gleichmäßiges Rauschen tolerierbar wäre?"

Klarstellung vorab (wichtig fürs Verständnis):
  In der aktuellen Pipeline VERWIRFT der composite-SQI keine Fenster — er ist
  ein WEICHES Fusionsgewicht in [0,1] (und optional ein Feature). Die hohe
  NaN-Rate kommt aus der HRV-Detektion (Problem 1), nicht aus dem SQI.
  ABER: ist das Rauschen über alle Fenster ähnlich, ist auch der SQI nahezu
  KONSTANT → gewichtete ≈ ungewichtete Fusion → der beobachtete Mini-Gewinn
  (+0.001) ist die ERWARTETE Folge, kein Fehler. Genau das prüft dieses Modul.

Vier Werkzeuge:
  1. sqi_spread_report()      Streut der SQI überhaupt? (CV, Perzentile,
                              effektive Gewichtsspanne → Decken-Abschätzung
                              für den Fusionsgewinn).
  2. validate_sqi_vs_hr_error()  GT-KONFORM (Goldstandard nur zur Validierung):
                              trennt hoher SQI kleinen HR-Fehler? -> ROC-AUC,
                              Spearman, Median-Fehler je SQI-Tertil.
  3. calibrate_min_sqi()      optimale SQI-Schwelle (Youden) auf der
                              SQI→"HR korrekt"-ROC; plus Gewicht-Tuning der
                              composite-Komponenten gegen das GT.
  4. simulate_artifact_robustness()  zeigt — auch bei sonst gleichmäßigem
                              Rauschen — dass der SQI gezielt EINGESTREUTE
                              Artefakte abwertet (der eigentliche Sinn fürs
                              Fusions-Argument der Arbeit).

Alle GT-Aufrufe dienen NUR der Validierung/Kalibrierung, nie der Berechnung
im Feature-/Fusionspfad (konsistent mit sqi.py).
"""

from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr

import oldCode.src.sqi as Q


# ──────────────────────────────────────────────────────────────────────────
# 1. Streut der SQI? (ohne GT)
# ──────────────────────────────────────────────────────────────────────────

def sqi_spread_report(sqi_values, labels=None, verbose=True):
    """
    Prüft, ob der SQI genug VARIIERT, um in der Fusion etwas bewirken zu können.
    Ist der SQI nahezu konstant (gleichmäßiges Rauschen), kann eine
    SQI-Gewichtung per Konstruktion fast nichts ändern.

    labels : optionale 0/1-Klassen (AF/Non-AF) -> getrennte Statistik.
    Rückgabe: dict mit Spannweite, CV, effektiver Gewichtsspanne und einer
    Decken-Abschätzung des relativen Einflusses auf das Fusionsmittel.
    """
    s = np.asarray(sqi_values, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 2:
        return {'n': len(s), 'informativ': False, 'grund': 'zu wenige Werte'}

    mean, std = float(s.mean()), float(s.std())
    cv = std / mean if mean > 0 else np.inf
    p05, p50, p95 = np.percentile(s, [5, 50, 95])
    # Effektive Gewichtsspanne: max. relativer Hebel eines Fensters im
    # gewichteten Mittel = (max-min)/mean. Klein -> Gewichtung ~ wirkungslos.
    eff = (s.max() - s.min()) / mean if mean > 0 else 0.0
    informativ = (cv >= 0.10) and (eff >= 0.30)

    res = {
        'n': int(len(s)), 'mean': mean, 'std': std, 'cv': float(cv),
        'p05': float(p05), 'p50': float(p50), 'p95': float(p95),
        'spannweite': float(s.max() - s.min()),
        'eff_gewichtsspanne': float(eff),
        'informativ': bool(informativ),
    }
    if labels is not None:
        labels = np.asarray(labels)[np.isfinite(np.asarray(sqi_values, float))]
        for c in (0, 1):
            sc = s[labels == c]
            if len(sc):
                res[f'mean_klasse{c}'] = float(sc.mean())

    if verbose:
        print(f"SQI-Streuung (n={res['n']}): mean={mean:.3f} std={std:.3f} "
              f"CV={cv:.2f}")
        print(f"  Perzentile  P05={p05:.3f}  P50={p50:.3f}  P95={p95:.3f}")
        print(f"  eff. Gewichtsspanne (max-min)/mean = {eff:.2f}")
        if informativ:
            print("  → SQI streut genug, um in der Fusion wirken zu können.")
        else:
            print("  → SQI nahezu KONSTANT: gewichtete ≈ ungewichtete Fusion. "
                  "Ein minimaler Gewinn ist hier ERWARTBAR (kein Bug). "
                  "Nutzen über simulate_artifact_robustness() zeigen.")
    return res


# ──────────────────────────────────────────────────────────────────────────
# 2. GT-Validierung: trennt hoher SQI kleinen HR-Fehler?
# ──────────────────────────────────────────────────────────────────────────

def _roc_auc(score, binary):
    """AUC ohne sklearn (Mann-Whitney-U). score höher -> eher binary==1."""
    score = np.asarray(score, float); binary = np.asarray(binary, int)
    pos = score[binary == 1]; neg = score[binary == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    r_pos = ranks[binary == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def validate_sqi_vs_hr_error(sqi_values, hr_est, hr_gt, tol_bpm=10.0, verbose=True):
    """
    GT-KONFORME Validierung (Goldstandard nur zur Bewertung):
    Definiert "Fenster korrekt" := |hr_est − hr_gt| ≤ tol_bpm und prüft, ob
    ein höherer SQI korrekte Fenster bevorzugt.

    hr_est : GT-freie HR-Schätzung je Fenster [bpm] (z.B. Q.estimate_hr_fft)
    hr_gt  : Referenz-HR aus dem GT-EKG je Fenster [bpm]

    Rückgabe: AUC (SQI→korrekt), Spearman(SQI, −|Fehler|), Median-|Fehler|
    je SQI-Tertil. AUC≈0.5 -> SQI nicht informativ für HR-Genauigkeit.
    """
    s = np.asarray(sqi_values, float)
    err = np.abs(np.asarray(hr_est, float) - np.asarray(hr_gt, float))
    ok = np.isfinite(s) & np.isfinite(err)
    s, err = s[ok], err[ok]
    if len(s) < 10:
        return {'n': len(s), 'auc': np.nan, 'grund': 'zu wenige gültige Fenster'}

    correct = (err <= tol_bpm).astype(int)
    auc = _roc_auc(s, correct)
    rho = float(spearmanr(s, -err).correlation)

    # Median-Fehler je SQI-Tertil
    q1, q2 = np.percentile(s, [33, 66])
    tert = {'niedrig': err[s <= q1], 'mittel': err[(s > q1) & (s <= q2)],
            'hoch': err[s > q2]}
    med = {k: (float(np.median(v)) if len(v) else np.nan) for k, v in tert.items()}

    res = {'n': int(len(s)), 'anteil_korrekt': float(correct.mean()),
           'auc': auc, 'spearman': rho, 'median_fehler_tertil': med}
    if verbose:
        print(f"SQI-Validierung gegen GT (n={res['n']}, tol={tol_bpm:.0f} bpm):")
        print(f"  korrekte Fenster gesamt: {res['anteil_korrekt']*100:.1f} %")
        print(f"  AUC (SQI→korrekt)      : {auc:.3f}  "
              f"({'informativ' if auc and auc > 0.6 else 'schwach/uninformativ'})")
        print(f"  Spearman(SQI,−|Fehler|): {rho:+.3f}")
        print(f"  Median |ΔHR| je Tertil : niedrig={med['niedrig']:.1f}  "
              f"mittel={med['mittel']:.1f}  hoch={med['hoch']:.1f} bpm")
    return res


# ──────────────────────────────────────────────────────────────────────────
# 3. Kalibrierung: Schwelle + Komponenten-Gewichte
# ──────────────────────────────────────────────────────────────────────────

def calibrate_min_sqi(sqi_values, hr_est, hr_gt, tol_bpm=10.0, verbose=True):
    """
    Wählt eine SQI-Schwelle (für min_sqi in der Fusion) per Youden-Index auf
    der SQI→"HR korrekt"-ROC. Liefert Schwelle + erreichte Sens/Spez.
    Hinweis: in der weichen Fusion ist min_sqi nur eine Abschneidegrenze für
    sehr schlechte Fenster — konservativ (niedrig) wählen.
    """
    s = np.asarray(sqi_values, float)
    err = np.abs(np.asarray(hr_est, float) - np.asarray(hr_gt, float))
    ok = np.isfinite(s) & np.isfinite(err)
    s, err = s[ok], err[ok]
    if len(s) < 10:
        return {'n': len(s), 'schwelle': 0.0, 'grund': 'zu wenige Fenster'}
    correct = (err <= tol_bpm).astype(int)

    best = {'schwelle': 0.0, 'youden': -1, 'sens': 0, 'spez': 0}
    for t in np.unique(np.round(s, 3)):
        keep = s >= t
        if keep.sum() == 0 or (~keep).sum() == 0:
            continue
        sens = correct[keep].mean()                       # behaltene sind korrekt
        spez = (1 - correct[~keep]).mean()                # verworfene waren falsch
        j = sens + spez - 1
        if j > best['youden']:
            best = {'schwelle': float(t), 'youden': float(j),
                    'sens': float(sens), 'spez': float(spez)}
    if verbose:
        print(f"Kalibrierte min_sqi-Schwelle: {best['schwelle']:.3f} "
              f"(Youden={best['youden']:.2f}, behaltene korrekt={best['sens']:.2f}, "
              f"verworfene falsch={best['spez']:.2f})")
    return best


def tune_composite_weights(signals, fs, hr_est, hr_gt, signal_type='ppg',
                           tol_bpm=10.0, step=0.25, verbose=True):
    """
    Sucht composite-Gewichte (w_t, w_p, w_k) per Gitter, die die SQI→korrekt-AUC
    maximieren. Komponenten (tSQI, pSQI, kurtosis-Qualität) werden EINMAL je
    Fenster berechnet und neu gewichtet — schnell.

    signals : Liste der Fenster (1D-Arrays), aligned zu hr_est/hr_gt.
    Rückgabe: beste Gewichte + AUC vs. AUC der Default-Gewichte (0.5,0.3,0.2).
    """
    band = Q.SQI_BANDS.get(signal_type, (0.6, 3.6))
    t = np.array([Q.tsqi(w, fs) for w in signals])
    p = np.array([Q.psqi(w, fs, band) for w in signals])
    k = np.array([Q._kurtosis_quality(Q.ksqi(w)) for w in signals])
    err = np.abs(np.asarray(hr_est, float) - np.asarray(hr_gt, float))
    ok = np.isfinite(err) & np.isfinite(t) & np.isfinite(p) & np.isfinite(k)
    t, p, k, err = t[ok], p[ok], k[ok], err[ok]
    correct = (err <= tol_bpm).astype(int)
    if len(correct) < 10 or correct.sum() in (0, len(correct)):
        if verbose:
            print("Gewicht-Tuning: nicht möglich (zu wenige Fenster oder alle "
                  "Fenster gleich korrekt/falsch — kein Trennsignal).")
        return {'grund': 'zu wenige/entartete Fenster', 'best': None}

    grid = np.arange(0, 1 + 1e-9, step)
    best = {'weights': (0.5, 0.3, 0.2), 'auc': -1}
    for wt in grid:
        for wp in grid:
            wk = 1 - wt - wp
            if wk < -1e-9 or wk > 1 + 1e-9:
                continue
            comp = wt * t + wp * p + wk * k
            auc = _roc_auc(comp, correct)
            if auc is not None and np.isfinite(auc) and auc > best['auc']:
                best = {'weights': (float(round(wt, 2)), float(round(wp, 2)),
                                    float(round(wk, 2))),
                        'auc': float(auc)}
    auc_default = _roc_auc(0.5 * t + 0.3 * p + 0.2 * k, correct)
    res = {'best_weights': best['weights'], 'best_auc': best['auc'],
           'default_auc': float(auc_default)}
    if verbose:
        print(f"Gewicht-Tuning (w_t,w_p,w_k): default {(0.5,0.3,0.2)} "
              f"AUC={auc_default:.3f}  ->  best {best['weights']} "
              f"AUC={best['auc']:.3f}")
        if best['auc'] <= auc_default + 0.01:
            print("  → kaum Verbesserung: Default-Gewichte beibehalten.")
    return res


# ──────────────────────────────────────────────────────────────────────────
# 4. Artefakt-Robustheit demonstrieren (der eigentliche SQI-Nutzen)
# ──────────────────────────────────────────────────────────────────────────

def simulate_artifact_robustness(clean_windows, fs, signal_type='ppg',
                                 frac_corrupt=0.3, severity=6.0, seed=0,
                                 verbose=True):
    """
    Zeigt: auch wenn der Datensatz sonst gleichmäßig verrauscht ist, WERTET der
    composite-SQI gezielt eingestreute Bewegungsartefakte ab. Das ist das
    tragfähige Argument fürs Fusions-Kapitel ("Robustheit gegenüber Artefakten").

    Nimmt saubere(re) Fenster, verfälscht einen Anteil mit Spike-Artefakten und
    vergleicht die SQI-Verteilung sauber vs. korrumpiert (Trenn-AUC).
    """
    rng = np.random.default_rng(seed)
    band = Q.SQI_BANDS.get(signal_type, (0.6, 3.6))
    sqi_clean, sqi_dirty = [], []
    for i, w in enumerate(clean_windows):
        w = np.asarray(w, float)
        sqi_clean.append(Q.composite_sqi(w, fs, band))
        if rng.random() < frac_corrupt:                 # Artefakt einstreuen
            wc = w.copy()
            for _ in range(rng.integers(3, 8)):
                j = rng.integers(0, len(wc))
                seg = slice(j, min(len(wc), j + int(0.2 * fs)))
                wc[seg] += rng.normal(0, severity * (np.std(w) + 1e-9),
                                      size=len(wc[seg]))
            sqi_dirty.append(Q.composite_sqi(wc, fs, band))
    sqi_clean = np.array(sqi_clean); sqi_dirty = np.array(sqi_dirty)
    scores = np.concatenate([sqi_clean, sqi_dirty])
    is_clean = np.concatenate([np.ones(len(sqi_clean)), np.zeros(len(sqi_dirty))])
    auc = _roc_auc(scores, is_clean.astype(int))
    res = {'sqi_sauber_median': float(np.median(sqi_clean)),
           'sqi_artefakt_median': float(np.median(sqi_dirty)),
           'trenn_auc': auc, 'n_artefakt': int(len(sqi_dirty))}
    if verbose:
        print(f"Artefakt-Robustheit ({signal_type}, {len(sqi_dirty)} korrumpiert):")
        print(f"  composite-SQI  sauber={res['sqi_sauber_median']:.3f}  "
              f"artefakt={res['sqi_artefakt_median']:.3f}")
        print(f"  Trenn-AUC (SQI erkennt Artefakt): {auc:.3f}  "
              f"({'gut' if auc and auc > 0.7 else 'schwach'})")
    return res


# ──────────────────────────────────────────────────────────────────────────
# Selbsttest
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    fs = 128
    rng = np.random.default_rng(0)
    t = np.arange(0, 30, 1 / fs)

    def ppg(bpm, noise, seed):
        r = np.random.default_rng(seed)
        f = bpm / 60.0
        return (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
                + noise * r.standard_normal(len(t)))

    # Szenario A: GEMISCHTE Qualität -> SQI sollte streuen & trennen.
    # HR-Schätzer = peak-basiert (versagt unter Rauschen) -> echte Fehler.
    print("══ Szenario A: gemischte Rauschstärke ══")
    sigs_A, hr_est_A, hr_gt_A, sqi_A = [], [], [], []
    for i in range(80):
        noise = rng.choice([0.1, 0.4, 1.5, 4.0])           # variabel
        bpm = rng.uniform(60, 90)
        w = ppg(bpm, noise, i)
        sigs_A.append(w)
        sqi_A.append(Q.composite_sqi(w, fs, Q.SQI_BANDS['ppg']))
        hr_gt_A.append(bpm)
        hr_est_A.append(Q.estimate_hr_peaks(w, fs))        # rauschempfindlich
    sqi_spread_report(sqi_A)
    print()
    validate_sqi_vs_hr_error(sqi_A, hr_est_A, hr_gt_A)
    print()
    calibrate_min_sqi(sqi_A, hr_est_A, hr_gt_A)
    print()
    tune_composite_weights(sigs_A, fs, hr_est_A, hr_gt_A, 'ppg')

    # Szenario B: GLEICHMÄSSIGES Rauschen -> SQI ~ konstant (dein Fall)
    print("\n══ Szenario B: gleichmäßiges Rauschen (SQI ~ konstant) ══")
    sqi_B = []
    for i in range(60):
        w = ppg(rng.uniform(60, 90), 1.0, 1000 + i)        # immer gleiches Rauschen
        sqi_B.append(Q.composite_sqi(w, fs, Q.SQI_BANDS['ppg']))
    sqi_spread_report(sqi_B)

    # Szenario C: Artefakt-Robustheit trotz gleichmäßigem Grundrauschen
    print("\n══ Szenario C: Artefakt-Robustheit ══")
    clean = [ppg(rng.uniform(60, 90), 0.3, 2000 + i) for i in range(60)]
    simulate_artifact_robustness(clean, fs, 'ppg', frac_corrupt=0.4)

    print("\n✓ Selbsttest erfolgreich.")
