"""
fusion_cv.py — Leckagefreie LOPO-Auswertung der SQI-gated Mixture of Experts (B)
================================================================================
Bachelorarbeit: AF-Detektion in kontaktlosen Signalen · Nik Büttner · RWTH Aachen

Das hier ist die Klammer um alles: Experten (experts.py) + Zuverlässigkeits-Ziel
(reliability.py) + Gate & Fusion (gating.py) + leckagefreie Auswertung (models.py).

Ablauf pro äußerem Fold (LeaveOneGroupOut über Patienten)
---------------------------------------------------------
Für jeden Test-Patienten (Rest = Trainingspatienten):

  1.  OOF-Experten-Wahrscheinlichkeiten auf den TRAININGS-Fenstern erzeugen
      (experts.oof_expert_probs) — ehrliche, nicht überoptimistische Outputs.
  2.  Gate-Ziel aus dem GT-EKG bauen (reliability-Tabelle). Ungültige Fenster
      (kein RR) -> hoher Fehler-Sentinel aus der TRAININGS-Verteilung
      (= "maximal unzuverlässig" -> Gewicht ~0).
  3.  Gate fitten:  SQI_train -> Fehler_train.   (Das GT-Ziel ist NUR hier nötig.)
  4.  Schwelle leckagefrei wählen: Gate-Gewichte x OOF-Probs -> fusionierte
      Trainings-Probs -> models.choose_threshold (Spez >= Ziel).
  5.  Experten auf ALLEN Trainingspatienten neu fitten, auf den Test-Patienten
      anwenden. Gate sagt aus dessen SQI die Gewichte vorher (KEIN GT im Test!).
      Fusion -> fensterweise AF-Wahrscheinlichkeit -> Schwelle -> Entscheidung.

Leakage-Kontrolle auf beiden Ebenen:
  * Patient: kein Patient gleichzeitig in Train und Test (LOPO, group=patient).
  * Stacking: Gate & Schwelle sehen nur OOF-Experten-Outputs, nie In-Sample-Probs.
  * GT: das GT-EKG fließt nur ins Trainings-Ziel des Gates, nie in den Test-Pfad.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

import experts as X
import gating as G
import models as M
import extract as E


# ──────────────────────────────────────────────────────────────────────────
# Hilfen: Reliability an die Feature-Tabelle ausrichten, Gate-Ziel vorbereiten
# ──────────────────────────────────────────────────────────────────────────

def align_reliability(df: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    """rel-Tabelle zeilengleich zu df anordnen (join über patient + win_idx)."""
    key = ['patient', 'win_idx']
    return df[key].merge(rel, on=key, how='left').reset_index(drop=True)


def _prep_targets(rel_train: pd.DataFrame, target_metric: str = 'target'):
    """
    Gate-Ziel (n, 3) in Reihenfolge G.ORDER. Ungültige (NaN) Fenster werden mit
    einem hohen Fehler-Sentinel aus der TRAININGS-Verteilung gefüllt
    (95-%-Quantil je Modalität) — invalide = maximal unzuverlässig.
    """
    cols = [f'rel_{m}_{target_metric}' for m in G.ORDER]
    T = rel_train[cols].values.astype(float)
    with np.errstate(all='ignore'):
        sentinel = np.nanpercentile(np.where(np.isfinite(T), T, np.nan), 95, axis=0)
    sentinel = np.where(np.isfinite(sentinel), sentinel, 1.0)
    T = np.where(np.isfinite(T), T, sentinel)
    return T


# ──────────────────────────────────────────────────────────────────────────
# Hauptauswertung
# ──────────────────────────────────────────────────────────────────────────

def evaluate_moe(df: pd.DataFrame, rel: pd.DataFrame, y, groups,
                 clf_per_modality: dict | None = None, gate_kind: str = 'mlp',
                 target_metric: str = 'target', inner_splits: int = 5,
                 min_spec: float = 0.80, random_state: int = 42,
                 return_arrays: bool = False):
    """
    Fensterweise AF-Auswertung der Mixture of Experts in patientenweiser LOPO-CV.

    gate_kind : 'mlp' | 'gb' | 'ridge'  -> gelerntes Gate (B)
                'equal'                  -> Gleichgewichts-Baseline (naive Fusion)
    Rückgabe  : (metrics_dict, mean_threshold[, y_true, y_prob, y_pred])
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    rel_al = align_reliability(df, rel)
    gate_cols = E.gate_sqi_cols(df, 'all')
    logo = LeaveOneGroupOut()

    yt, yp, yd, yg, used_t = [], [], [], [], []
    for tr, te in logo.split(df, y, groups):
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        y_tr, g_tr = y[tr], groups[tr]

        # 1) leckagefreie OOF-Experten-Probs auf den Trainingsfenstern
        oof_tr = X.oof_expert_probs(df_tr, y_tr, g_tr, clf_per_modality,
                                    n_splits=inner_splits, random_state=random_state)
        P_tr = G.probs_matrix(oof_tr)

        # 3/4) Gate fitten + Trainingsgewichte + Schwelle
        if gate_kind == 'equal':
            w_tr = G.equal_weights(len(df_tr))
            gate, scale = None, None
        else:
            T_tr = _prep_targets(rel_al.iloc[tr], target_metric)
            gate = G.make_gate(kind=gate_kind, random_state=random_state)
            gate.fit(df_tr[gate_cols].values, T_tr)
            err_tr = gate.predict(df_tr[gate_cols].values)
            scale = float(np.nanmedian(err_tr)) + 1e-6   # feste Temperatur aus Training
            w_tr = G.errors_to_weights(err_tr, scale=scale)

        fused_tr = G.fuse(w_tr, P_tr)
        t = M.choose_threshold(y_tr, fused_tr, min_spec)

        # 5) Experten auf ALLEN Trainingspatienten neu fitten, auf Test anwenden
        experts = X.fit_experts(df_tr, y_tr, clf_per_modality, random_state=random_state)
        P_te = G.probs_matrix(X.expert_probs(experts, df_te))
        if gate_kind == 'equal':
            w_te = G.equal_weights(len(df_te))
        else:
            w_te = G.errors_to_weights(gate.predict(df_te[gate_cols].values), scale=scale)
        fused_te = G.fuse(w_te, P_te)

        yt.extend(y[te]); yp.extend(fused_te); yd.extend((fused_te >= t).astype(int))
        yg.extend(groups[te]); used_t.append(t)

    yt, yp, yd = map(np.array, (yt, yp, yd))
    yg = np.array(yg)
    m = M.metrics(yt, yp, yd)
    out = (m, float(np.mean(used_t)))
    if return_arrays:
        out = out + (yt, yp, yd, yg)
    return out


def compare_gates(df: pd.DataFrame, rel: pd.DataFrame, y, groups,
                  gate_kinds=('equal', 'ridge', 'gb', 'mlp'),
                  clf_per_modality: dict | None = None, **kw) -> pd.DataFrame:
    """
    Vergleicht naive (equal) gegen gelernte Gates — das zentrale Argument der
    Arbeit: bringt das datengetriebene Gating überhaupt etwas gegenüber
    Gleichgewichts-Fusion?
    """
    rows = []
    for gk in gate_kinds:
        m, t = evaluate_moe(df, rel, y, groups, clf_per_modality=clf_per_modality,
                            gate_kind=gk, **kw)
        rows.append({'gate': gk, **m, 'threshold': round(t, 3)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Selbsttest: voller leckagefreier Lauf auf synthetischen Daten
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    SQI_KEYS = ['kSQI', 'sSQI', 'pSQI', 'bSQI', 'tSQI']
    SIGS = {'cecg': ['cecg'], 'ppg': ['ppg1', 'ppg2'], 'bcg': ['bcg1', 'bcg2']}
    ALL_SIGS = ['cecg', 'ppg1', 'ppg2', 'bcg1', 'bcg2']
    STRENGTH = {'cecg': 0.7, 'ppg1': 1.5, 'ppg2': 1.5, 'bcg1': 0.1, 'bcg2': 0.1}

    feat_rows, rel_rows = [], []
    for p in range(14):
        af = int(p % 2 == 0)
        for w in range(18):
            sqi = {s: {k: rng.uniform(0, 1) for k in SQI_KEYS} for s in ALL_SIGS}
            r = {'patient': f'PAT{p:03d}', 'AF': af, 'win_idx': w}
            for s in ALL_SIGS:
                for k in range(5):
                    r[f'{s}_f{k}'] = STRENGTH[s] * (af - 0.5) + rng.standard_normal()
                for k in SQI_KEYS:
                    r[f'sqi_{s}_{k}'] = sqi[s][k]
            feat_rows.append(r)
            # Reliability-Ziel: PPG klein, BCG groß; hängt (negativ) von pSQI ab,
            # damit das Gate aus dem SQI etwas lernen kann.
            rel_rows.append({
                'patient': f'PAT{p:03d}', 'win_idx': w,
                'rel_cecg_target': 1.0 - 0.6 * sqi['cecg']['pSQI'] + 0.1 * rng.standard_normal(),
                'rel_ppg_target':  0.4 - 0.3 * sqi['ppg1']['pSQI'] + 0.1 * rng.standard_normal(),
                'rel_bcg_target':  2.0 - 0.4 * sqi['bcg1']['pSQI'] + 0.1 * rng.standard_normal(),
            })
    df = pd.DataFrame(feat_rows)
    rel = pd.DataFrame(rel_rows)
    y = df['AF'].values
    groups = df['patient'].values
    print('Synthetik:', df.shape, '· Gate-SQI-Eingänge:', len(E.gate_sqi_cols(df, 'all')))

    tab = compare_gates(df, rel, y, groups, gate_kinds=('equal', 'ridge', 'gb'),
                        inner_splits=3)
    cols = ['gate', 'AUC', 'Sensitivität', 'Spezifität', 'Accuracy', 'threshold']
    print('\n', tab[[c for c in cols if c in tab.columns]].round(3).to_string(index=False))
    print('\nSelbsttest OK (gelerntes Gate sollte BCG-Rauschen abwerten und equal schlagen).')
