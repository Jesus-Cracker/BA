"""
gating.py — Gating-Netz + gewichtete Fusion (Track B, Bachelet-Stil)
====================================================================
Bachelorarbeit: AF-Detektion in kontaktlosen Signalen · Nik Büttner · RWTH Aachen

Aufgabe
-------
Das Gating-Netz sagt aus den GT-FREIEN SQIs eines Fensters den
Zuverlässigkeits-FEHLER jeder Modalität voraus (Ziel kommt aus reliability.py,
GT-EKG-basiert — nur im Training verfügbar). Aus den prädizierten Fehlern werden
Gewichte: kleiner Fehler -> großes Gewicht. Die gewichtete Summe der drei
Experten-Wahrscheinlichkeiten ergibt die finale AF-Wahrscheinlichkeit.

    SQI ── Gate ──> Fehler_hat (3)  ──Abbildung──>  Gewichte (3, Σ=1)
                                                        │
            p_cecg, p_ppg, p_bcg  ────────────────────►  Σ wₖ·pₖ  ──► AF

Das entspricht Bachelets "Fehlerprädiktion mittels eines KNN" (3.6.1) plus
"Abbildung des prädizierten Fehlers auf einen SQI / gewichtete Fusion" (3.6.2/3.6.3).

Hinweise
--------
* Der Gate-Regressor ist austauschbar (`kind`): 'mlp' (KNN, nah an Bachelet),
  'gb' (HistGradientBoosting, robust & schnell), 'ridge' (linearer Sanity-Baseline).
* Die Reihenfolge der Modalitäten ist FEST (ORDER) — Gewichte und
  Wahrscheinlichkeiten müssen in derselben Reihenfolge vorliegen.
* Ein-/Ausgaben werden normalisiert (StandardScaler innen + skaliertes Ziel),
  sonst dominieren großskalige Größen das Lernen (vgl. Bachelet 3.6.1).
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

ORDER = ['cecg', 'ppg', 'bcg']   # feste Modalitäts-Reihenfolge


def make_gate(kind: str = 'mlp', hidden=(64, 32), alpha: float = 1e-3,
              random_state: int = 42):
    """
    Multi-Output-Regressor: SQI-Vektor -> [err_cecg, err_ppg, err_bcg].
    Ein- und Ausgaben werden normalisiert (Ziel via TransformedTargetRegressor).
    """
    if kind == 'mlp':
        from sklearn.neural_network import MLPRegressor
        reg = MLPRegressor(hidden_layer_sizes=hidden, activation='relu',
                           alpha=alpha, max_iter=800, random_state=random_state)
    elif kind == 'gb':
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        reg = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=random_state))
    elif kind == 'ridge':
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0, random_state=random_state)
    else:
        raise ValueError("kind muss 'mlp', 'gb' oder 'ridge' sein")

    inner = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc',  StandardScaler()),
                      ('reg', reg)])
    # Ziel ebenfalls normalisieren (Fehlerwerte verschiedener Modalitäten skalieren unterschiedlich)
    return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())


def errors_to_weights(err_hat, scale: float | None = None, eps: float = 1e-6):
    """
    Monoton fallende Abbildung Fehler -> Gewicht, zeilenweise normiert (Σ=1).
    Softmax über den negierten, skalierten Fehler:  wₖ ∝ exp(-errₖ / scale).

    scale (Temperatur): None -> Median aller Fehler (datengetrieben). Kleiner
    scale = härtere Gewichtung (fast Argmin), großer scale = gleichmäßiger.
    """
    err = np.asarray(err_hat, dtype=float)
    if err.ndim == 1:
        err = err[None, :]
    if scale is None:
        scale = float(np.nanmedian(err)) + eps
    z = -err / (scale + eps)
    z = z - np.nanmax(z, axis=1, keepdims=True)        # numerisch stabil
    w = np.exp(np.nan_to_num(z, nan=-np.inf))
    w = w / (w.sum(axis=1, keepdims=True) + eps)
    return w


def equal_weights(n_rows: int, n_mod: int = 3):
    """Gleichgewichts-Baseline (naive Fusion) — für den Vergleich im Schreibteil."""
    return np.full((n_rows, n_mod), 1.0 / n_mod)


def fuse(weights, probs):
    """
    Gewichtete Fusion: weights (n, 3) · probs (n, 3) -> fused prob (n,).
    BEIDE müssen in der Reihenfolge ORDER vorliegen.
    """
    weights = np.asarray(weights, float)
    probs = np.asarray(probs, float)
    return (weights * probs).sum(axis=1)


def probs_matrix(prob_df, order=ORDER):
    """Holt die Spalten p_<mod> als (n, 3)-Matrix in fester Reihenfolge."""
    return prob_df[[f'p_{m}' for m in order]].values


# ──────────────────────────────────────────────────────────────────────────
# Selbsttest: Gate lernt SQI->Fehler, niedriger Fehler -> höheres Gewicht
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n = 600
    SQI = rng.uniform(0, 1, size=(n, 25))           # 25 SQI-Eingänge (5 Signale x 5)

    # Wahres Ziel: Fehler hängt (negativ) von bestimmten SQIs ab + Rauschen.
    # PPG-Fehler systematisch klein, BCG-Fehler systematisch groß.
    err_true = np.column_stack([
        1.0 - 0.8 * SQI[:, 0]  + 0.1 * rng.standard_normal(n),   # cECG mittel
        0.4 - 0.3 * SQI[:, 5]  + 0.1 * rng.standard_normal(n),   # PPG  klein
        2.0 - 0.5 * SQI[:, 15] + 0.1 * rng.standard_normal(n),   # BCG  groß
    ])

    split = 450
    gate = make_gate(kind='gb')
    gate.fit(SQI[:split], err_true[:split])
    err_hat = gate.predict(SQI[split:])

    from sklearn.metrics import r2_score
    print('Gate R² je Modalität:',
          {m: round(r2_score(err_true[split:, i], err_hat[:, i]), 3) for i, m in enumerate(ORDER)})

    w = errors_to_weights(err_hat)
    print('Gewichtssummen ~1 :', np.allclose(w.sum(axis=1), 1.0))
    print('mittlere Gewichte :', {m: round(w[:, i].mean(), 3) for i, m in enumerate(ORDER)})
    assert w[:, 1].mean() > w[:, 2].mean(), 'PPG (kleiner Fehler) muss im Mittel höher gewichtet sein als BCG'

    # Fusion-Sanity
    probs = rng.uniform(0, 1, size=(len(w), 3))
    fused = fuse(w, probs)
    print('fused shape       :', fused.shape, '· in [0,1]:', bool((fused >= 0).all() and (fused <= 1).all()))
    print('Selbsttest OK.')
