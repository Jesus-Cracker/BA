"""
experts.py — Drei modalitätsspezifische Experten + leckagefreie OOF-Probs
=========================================================================
Bachelorarbeit: AF-Detektion in kontaktlosen Signalen · Nik Büttner · RWTH Aachen
Track B: SQI-gated Mixture of Experts (Bachelet-Stil)

Rolle in der Architektur
------------------------
Jeder Experte ist ein eigenständiger AF-Klassifikator für GENAU EINE Modalität:

    cECG-Experte : nur 'cecg_*'-Merkmale          -> p_cecg(AF | Fenster)
    PPG-Experte  : 'ppg1_*' + 'ppg2_*'-Merkmale   -> p_ppg (AF | Fenster)
    BCG-Experte  : 'bcg1_*' + 'bcg2_*'-Merkmale   -> p_bcg (AF | Fenster)

Es wird die bestehende, validierte `models.make_pipelines` wiederverwendet
(Imputer + Scaler + RF-Feature-Selektion IN der Pipeline → pro Fold neu gefittet,
also kein Leakage innerhalb des Experten).

DER ENTSCHEIDENDE PUNKT: `oof_expert_probs`
-------------------------------------------
Das Gate (nächstes Modul) lernt aus den Ausgaben der Experten. Würde es mit
Wahrscheinlichkeiten trainiert, die ein Experte für seine EIGENEN Trainingsfenster
ausgibt, wären diese zu optimistisch (der Experte hat die Fenster ja gesehen) →
das Gate lernte einen verzerrten Zusammenhang und die Endmetriken wären geschönt.

`oof_expert_probs` erzeugt deshalb **Out-of-Fold-Wahrscheinlichkeiten**: jeder
Experte sagt nur Fenster vorher, die NICHT in seinem Training waren. Gruppiert wird
nach Patient (StratifiedGroupKFold), damit auch hier kein Patient gleichzeitig in
Train und Val liegt. Das ist dieselbe Idee wie beim leckagefreien Threshold-Tuning
in `models.evaluate_*_nested` — nur eben für die Experten-Outputs.

Hinweis: Dieses Modul ist unabhängig von der A/B-Entscheidung. Die drei Experten und
ihre OOF-Probs werden in beiden Varianten gebraucht.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

import models as M
import extract as E


# Standard-Klassifikator je Experte (LR = robuster, gut kalibrierter Default;
# pro Modalität frei wählbar, z.B. {'cecg':'RF','ppg':'LR','bcg':'GB'}).
DEFAULT_CLF = {'cecg': 'LR', 'ppg': 'LR', 'bcg': 'LR'}


def _scale_pos_weight(y) -> float:
    """n_negativ / n_positiv — für XGBoost-Klassengewichtung (sonst 1.0)."""
    y = np.asarray(y)
    n_pos = max(int((y == 1).sum()), 1)
    return float((y == 0).sum()) / n_pos


def build_expert(modality: str, clf='LR', balanced: bool = True,
                 scale_pos_weight: float = 1.0, random_state: int = 42):
    """Eine frische (ungefittete) Experten-Pipeline für eine Modalität.

    `clf` kann sein:
      * ein String, z.B. 'LR', 'XGB', 'SVM';
      * mit Suffix '+cal' (z.B. 'LR+cal', 'XGB+cal') -> isotone Wahrscheinlichkeits-
        Kalibrierung (CalibratedClassifierCV, cv=3) UM die Pipeline. Pro Fold
        INNERHALB der Trainingsdaten gefittet -> leckagefrei. Sinnvoll, weil die
        Fusion eine gewichtete SUMME der Probs ist: schlecht kalibrierte Experten
        verzerren Summe und Schwelle. 'SVM' ist bei balanced=True bereits isoton
        kalibriert -> '+cal' wird dort übersprungen;
      * ein Tupel (name, params) mit getunten Hyperparametern, z.B.
        ('LR+cal', {'C': 3.2, 'penalty': 'l1', 'solver': 'liblinear'}) oder
        ('XGB', {'max_depth': 4, 'learning_rate': 0.03}). Die Keys werden als
        clf__<key> an die Pipeline gesetzt (siehe tune_expert).

    WICHTIG (Cache): Da clf_per_modality in die Fold-Cache-Signatur eingeht
    (repr der Items), landen sowohl '+cal' als auch getunte Params automatisch im
    Hash — kalibrierte/getunte Varianten kollidieren NICHT mit alten Caches.
    """
    if isinstance(clf, (tuple, list)):
        clf_str, params = clf[0], dict(clf[1] or {})
    else:
        clf_str, params = clf, {}
    calibrate = clf_str.endswith('+cal')
    base = clf_str[:-4] if calibrate else clf_str
    pipes = M.make_pipelines(balanced=balanced, random_state=random_state,
                             scale_pos_weight=scale_pos_weight)
    if base not in pipes:
        raise ValueError(f"Klassifikator '{base}' nicht verfügbar. Wähle: {list(pipes)} "
                         f"(optionales Suffix '+cal' für isotone Kalibrierung).")
    pipe = clone(pipes[base])
    if params:   # getunte Hyperparameter VOR einer evtl. Kalibrierung setzen
        pipe.set_params(**{f'clf__{k}': v for k, v in params.items()})
    if calibrate and base != 'SVM':
        pipe = CalibratedClassifierCV(pipe, method='isotonic', cv=3)
    return pipe


def fit_experts(df: pd.DataFrame, y, clf_per_modality: dict | None = None,
                balanced: bool = True, random_state: int = 42) -> dict:
    """
    Trainiert die drei Experten auf ALLEN übergebenen Fenstern.
    Rückgabe: {modality: {'pipe', 'cols', 'clf'}}.
    (Für die finale Anwendung pro äußerem CV-Fold auf den Trainingspatienten.)
    """
    clf_per_modality = clf_per_modality or DEFAULT_CLF
    y = np.asarray(y)
    spw = _scale_pos_weight(y)
    experts = {}
    for m, clf in clf_per_modality.items():
        cols = E.expert_feature_cols(df, m)
        pipe = build_expert(m, clf, balanced, spw, random_state)
        pipe.fit(df[cols].values, y)
        experts[m] = {'pipe': pipe, 'cols': cols, 'clf': clf}
    return experts


def expert_probs(experts: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Fensterweise AF-Wahrscheinlichkeit je Experte: Spalten p_cecg / p_ppg / p_bcg."""
    out = {}
    for m, info in experts.items():
        out[f'p_{m}'] = info['pipe'].predict_proba(df[info['cols']].values)[:, 1]
    return pd.DataFrame(out, index=df.index)


def oof_expert_probs(df: pd.DataFrame, y, groups, clf_per_modality: dict | None = None,
                     balanced: bool = True, n_splits: int = 5,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Leckagefreie Out-of-Fold-Wahrscheinlichkeiten je Experte (Eingang fürs Gate).

    Jedes Fenster wird genau einmal vorhergesagt — von Experten, die auf ANDEREN
    Patienten trainiert wurden. Spalten: p_cecg / p_ppg / p_bcg (Index = df-Index).
    """
    clf_per_modality = clf_per_modality or DEFAULT_CLF
    y = np.asarray(y)
    groups = np.asarray(groups)
    spw = _scale_pos_weight(y)
    n = len(df)
    oof = {m: np.full(n, np.nan) for m in clf_per_modality}

    n_splits = min(n_splits, len(np.unique(groups)))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cols_by_mod = {m: E.expert_feature_cols(df, m) for m in clf_per_modality}
    for tr, va in skf.split(df, y, groups):
        for m, clf in clf_per_modality.items():
            cols = cols_by_mod[m]
            pipe = build_expert(m, clf, balanced, spw, random_state)
            pipe.fit(df.iloc[tr][cols].values, y[tr])
            oof[m][va] = pipe.predict_proba(df.iloc[va][cols].values)[:, 1]

    return pd.DataFrame({f'p_{m}': oof[m] for m in clf_per_modality}, index=df.index)


def tune_expert(df: pd.DataFrame, y, groups, modality: str, clf: str = 'LR',
                n_trials: int = 80, cv_splits: int = 5, random_state: int = 42,
                balanced: bool = True, n_jobs: int = 1, verbose: bool = True):
    """Optuna-Suche für EINEN Modalitäts-Experten — direkt aufs AF-Ziel.

    Zielgröße (direction='maximize') = gepoolte OOF-AUC dieser Modalität in
    PATIENTENGRUPPIERTER CV (StratifiedGroupKFold, identisch zu oof_expert_probs)
    -> die zurückgegebene AUC ist direkt mit den OOF-AUCs im Notebook vergleichbar
    und leckagefrei (Imputer/Skalierung/Feature-Selektion liegen IN der Pipeline,
    pro Fold neu gefittet; kein Patient gleichzeitig in Train/Val).

    `clf`: 'LR' oder 'XGB' (Basis, OHNE '+cal' — Kalibrierung erst beim Deployment).
    Suchräume:
      LR : C (log 1e-3..1e2), penalty {l1,l2} (solver=liblinear)
      XGB: n_estimators, max_depth, learning_rate(log), subsample,
           colsample_bytree, min_child_weight, reg_lambda(log)

    Rückgabe: (best_params, study). Einsatz direkt als getunter Experte über die
    Tupel-Form von clf_per_modality, z.B.:
        clf_per_modality = {'ppg': ('LR+cal', best_ppg), 'cecg': 'XGB+cal', 'bcg': 'XGB+cal'}
        fold_cache = CV.precompute_folds(df, y, groups, clf_per_modality=..., ...)
    (Die getunten Params landen automatisch in der Cache-Signatur.)
    """
    import optuna
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y); groups = np.asarray(groups)
    cols = E.expert_feature_cols(df, modality)
    Xm = df[cols].values
    spw = _scale_pos_weight(y)
    n_splits = min(cv_splits, len(np.unique(groups)))
    folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=random_state).split(df, y, groups))

    def _suggest(trial):
        if clf == 'LR':
            # nur C tunen: 'penalty' ist in sklearn 1.8 deprecated; die L1-Sparsity
            # übernimmt ohnehin die Feature-Selektion in der Pipeline.
            return {'C': trial.suggest_float('C', 1e-3, 1e2, log=True)}
        elif clf == 'XGB':
            return {'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                    'max_depth': trial.suggest_int('max_depth', 2, 6),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-2, 3e-1, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)}
        raise ValueError("tune_expert unterstützt 'LR' oder 'XGB' (Basis, ohne '+cal').")

    def objective(trial):
        params = _suggest(trial)
        oof = np.full(len(df), np.nan)
        for tr, va in folds:
            pipe = build_expert(modality, (clf, params), balanced, spw, random_state)
            pipe.fit(Xm[tr], y[tr])
            oof[va] = pipe.predict_proba(Xm[va])[:, 1]
        return roc_auc_score(y, oof)

    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    best = dict(study.best_params)
    if verbose:
        print(f'{modality} {clf}: beste OOF-AUC = {study.best_value:.3f} · Params: {best}')
    return best, study


# ──────────────────────────────────────────────────────────────────────────
# Selbsttest auf synthetischen Daten (kein echter Datensatz nötig)
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rng = np.random.default_rng(0)

    # Synthetische Feature-Tabelle: 12 Patienten (6 AF / 6 Non-AF), je 20 Fenster.
    # PPG trägt das Signal (informativ), cECG schwächer, BCG fast Rauschen
    # — soll später dem Gate erlauben, PPG hochzugewichten.
    sigs = {'cecg': ['cecg'], 'ppg': ['ppg1', 'ppg2'], 'bcg': ['bcg1', 'bcg2']}
    rows = []
    for p in range(12):
        af = int(p % 2 == 0)
        for w in range(20):
            r = {'patient': f'PAT{p:03d}', 'AF': af, 'win_idx': w}
            for s in ['cecg', 'ppg1', 'ppg2', 'bcg1', 'bcg2']:
                # informativ je nach Modalität unterschiedlich stark
                strength = {'cecg': 0.6, 'ppg1': 1.4, 'ppg2': 1.4, 'bcg1': 0.1, 'bcg2': 0.1}[s]
                for k in range(5):
                    r[f'{s}_f{k}'] = strength * (af - 0.5) + rng.standard_normal()
                r[f'sqi_{s}_composite'] = rng.uniform(0, 1)
            rows.append(r)
    df = pd.DataFrame(rows)
    df, y, groups = E.split_Xygroups(df)

    print('Synthetische Tabelle:', df.shape)
    for m in sigs:
        print(f'  Experte {m:5s}: {len(E.expert_feature_cols(df, m))} Merkmale')

    oof = oof_expert_probs(df, y, groups, n_splits=3)
    print('\nOOF-Probs:', oof.shape, '· NaN:', int(oof.isna().sum().sum()))
    # AUC je Experte auf den OOF-Probs (Plausibilität: PPG > cECG > BCG)
    from sklearn.metrics import roc_auc_score
    for c in oof.columns:
        print(f'  {c}: OOF-AUC = {roc_auc_score(y, oof[c]):.3f}')

    experts = fit_experts(df, y)
    probs = expert_probs(experts, df)
    print('\nfit_experts + expert_probs OK:', probs.shape)
    print('Selbsttest OK.')
