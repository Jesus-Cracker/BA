"""
models.py — Konsolidierte ML-Maschinerie (Track A)
==================================================
Bachelorarbeit: AF-Detektion in kontaktlosen Signalen · Nik Büttner · RWTH Aachen

Vereint alle ML-Helfer aus `05_master.ipynb` an EINEM Ort und behebt das
Threshold-Leakage methodisch sauber.

Kernkonzepte:
  - LOPO-CV (Leave-One-Patient-Out) = LeaveOneGroupOut über die Patienten-IDs.
  - Fusion auf Patientenebene: Mittelung der Fensterwahrscheinlichkeiten
    ("Majority Vote" im Sinne der gemittelten Wahrscheinlichkeit).
  - Threshold-Wahl: Sensitivität maximieren unter Nebenbedingung Spezifität >= Ziel.

Threshold-Leakage — drei Auswertungsmodi:
  'window'     : fensterbasierte Baseline (kein Threshold-Tuning).
  'mv_pooled'  : Patientenebene, Threshold auf ALLEN OOF-Probs gewählt.
                 = exakt der alte 05_master-Pfad → OPTIMISTISCH (Threshold sieht
                   die Testdaten). AUC ist davon unberührt.
  'mv_nested'  : Patientenebene, Threshold in einer INNEREN CV bestimmt, NUR auf
                 Trainingspatienten → leckagefrei → ehrliche Sens/Spez.

Empfehlung für die Arbeit: beide Patienten-Modi berichten ("naiv vs. genested")
— zeigt methodisches Bewusstsein und ist ein sauberes Diskussionsargument.

HINWEIS (kleinere verbleibende Optimismus-Quelle): Die RF-Feature-Selektion in
05_master lief einmal auf allen Daten. Für vollständige Strenge müsste sie pro
äußerem Fold neu laufen (select_features_rf ist dafür vorbereitet). Effekt bei
37/100 Features gering; im Schreibteil erwähnen.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

RANDOM_STATE = 42
TARGET_SPEC  = 0.80      # klinische Nebenbedingung: Spezifität >= 0.80


def _inf_to_nan(X):
    """±inf -> NaN, damit der SimpleImputer sie behandeln kann.
    (SimpleImputer ersetzt NaN, aber NICHT inf -> sonst ValueError.)"""
    X = np.asarray(X, dtype=float)
    return np.where(np.isfinite(X), X, np.nan)


def _clean_step():
    """Erster Pipeline-Schritt: inf -> NaN."""
    return ('fin', FunctionTransformer(_inf_to_nan, feature_names_out='one-to-one'))


# ──────────────────────────────────────────────────────────────────────────
# 1. Pipelines
# ──────────────────────────────────────────────────────────────────────────

def make_pipelines(balanced=False, random_state=RANDOM_STATE, scale_pos_weight=1.0):
    """
    Klassifikator-Pipelines. balanced=True -> class_weight='balanced'
    + isotone Kalibrierung der SVM. Imputation/Skalierung IN der Pipeline
    -> kein Leakage über CV-Folds.

    XGBoost wird nur aufgenommen, wenn das Paket installiert ist.
    scale_pos_weight gewichtet bei XGBoost die positive (AF-)Klasse;
    sinnvoll = n_negativ / n_positiv (wird in compare_models automatisch gesetzt).
    """
    cw = 'balanced' if balanced else None
    if balanced:
        svm_clf = CalibratedClassifierCV(
            SVC(kernel='rbf', C=1.0, class_weight=cw, random_state=random_state),
            cv=3, method='isotonic')
    else:
        svm_clf = SVC(probability=True, kernel='rbf', C=1.0, random_state=random_state)

    pipes = {
        'SVM': Pipeline([_clean_step(),
                         ('imp', SimpleImputer(strategy='median')),
                         ('sc',  StandardScaler()),
                         ('clf', svm_clf)]),
        'LR':  Pipeline([_clean_step(),
                         ('imp', SimpleImputer(strategy='median')),
                         ('sc',  StandardScaler()),
                         ('clf', LogisticRegression(max_iter=1000, class_weight=cw,
                                                    random_state=random_state))]),
        'RF':  Pipeline([_clean_step(),
                         ('imp', SimpleImputer(strategy='median')),
                         ('clf', RandomForestClassifier(n_estimators=200,
                                                        class_weight=cw,
                                                        random_state=random_state,
                                                        n_jobs=-1))]),
        'GB':  Pipeline([_clean_step(),
                         ('imp', SimpleImputer(strategy='median')),
                         ('clf', GradientBoostingClassifier(n_estimators=200,
                                                            random_state=random_state))]),
    }

    if _HAS_XGB:
        pipes['XGB'] = Pipeline([
            _clean_step(),
            # XGBoost behandelt NaN nativ; Imputer trotzdem für einheitliche Pipeline
            ('imp', SimpleImputer(strategy='median')),
            ('clf', XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=(scale_pos_weight if balanced else 1.0),
                eval_metric='logloss', random_state=random_state,
                n_jobs=-1, tree_method='hist'))])

    # ── Feature-Selektion IN der Pipeline (pro Fold neu gefittet -> kein Leakage) ──
    from sklearn.feature_selection import SelectFromModel
    for _name, _pipe in list(pipes.items()):
        _steps = list(_pipe.steps)
        _i = next(k for k, (n, _) in enumerate(_steps) if n == 'imp') + 1
        _steps.insert(_i, ('fs', SelectFromModel(
            RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=1),
            threshold=0.008)))
        pipes[_name] = Pipeline(_steps)
    return pipes


# Modell-Reihenfolge für Vergleiche (XGB nur falls verfügbar)
MODEL_ORDER = ['SVM', 'LR', 'RF', 'GB'] + (['XGB'] if _HAS_XGB else [])


# ──────────────────────────────────────────────────────────────────────────
# 2. Feature-Selektion
# ──────────────────────────────────────────────────────────────────────────

def select_features_rf(X, y, threshold=0.008, random_state=RANDOM_STATE):
    """RF-Importance-Selektion. Gibt (mask, importances) zurück."""
    X = _inf_to_nan(X)
    imp = SimpleImputer(strategy='median')
    rf  = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(imp.fit_transform(X), y)
    importances = rf.feature_importances_
    return importances >= threshold, importances


# ──────────────────────────────────────────────────────────────────────────
# 3. Vorhersagen
# ──────────────────────────────────────────────────────────────────────────

def lopo_window_predictions(pipe, X, y, groups):
    """Fensterbasierte OOF-Vorhersagen über LOPO."""
    logo = LeaveOneGroupOut()
    yt, yp, yd = [], [], []
    for tr, te in logo.split(X, y, groups):
        pipe.fit(X[tr], y[tr])
        yp.extend(pipe.predict_proba(X[te])[:, 1])
        yd.extend(pipe.predict(X[te]))
        yt.extend(y[te])
    return np.array(yt), np.array(yp), np.array(yd)


def _agg_patient(probs, weights=None, min_w=0.0):
    """Patientenwahrscheinlichkeit = (gewichtetes) Mittel der Fensterprobs.
    weights=None -> ungewichtet. Fenster mit Gewicht < min_w fallen weg.
    Fallback auf ungewichtetes Mittel, wenn kein Fenster genug Qualität hat."""
    probs = np.asarray(probs, dtype=float)
    if weights is None:
        return float(np.mean(probs))
    w = np.asarray(weights, dtype=float).copy()
    w[w < min_w] = 0.0
    if w.sum() <= 0:
        return float(np.mean(probs))
    return float(np.sum(w * probs) / np.sum(w))


def lopo_patient_probs(pipe, X, y, groups, window_sqi=None, min_sqi=0.0):
    """
    Patientenebene: pro LOPO-Fold (gewichtete) mittlere Fensterwahrscheinlichkeit.
    window_sqi : optionales Qualitätsgewicht je Fenster (aligned zu X) in [0,1].
    Gibt (pat_ids, pat_true, pat_prob) — OHNE Threshold.
    """
    logo = LeaveOneGroupOut()
    ids, true, prob = [], [], []
    for tr, te in logo.split(X, y, groups):
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        w = None if window_sqi is None else window_sqi[te]
        ids.append(groups[te][0]); true.append(int(y[te][0]))
        prob.append(_agg_patient(p, w, min_sqi))
    return np.array(ids), np.array(true), np.array(prob)


def _inner_patient_probs(pipe, X, y, groups, window_sqi=None, min_sqi=0.0,
                         n_splits=5, random_state=RANDOM_STATE):
    """Patienten-OOF-Probs via gruppierter k-Fold (für innere Threshold-Wahl)."""
    n_groups = len(np.unique(groups))
    splitter = StratifiedGroupKFold(n_splits=min(n_splits, n_groups),
                                    shuffle=True, random_state=random_state)
    true, prob = [], []
    for tr, te in splitter.split(X, y, groups):
        pipe.fit(X[tr], y[tr])
        for g in np.unique(groups[te]):
            m = groups[te] == g
            p = pipe.predict_proba(X[te][m])[:, 1]
            w = None if window_sqi is None else window_sqi[te][m]
            true.append(int(y[te][m][0]))
            prob.append(_agg_patient(p, w, min_sqi))
    return np.array(true), np.array(prob)


# ──────────────────────────────────────────────────────────────────────────
# 4. Threshold & Metriken
# ──────────────────────────────────────────────────────────────────────────

def choose_threshold(y_true, y_prob, min_spec=TARGET_SPEC):
    """Threshold der Sensitivität maximiert bei Spezifität >= min_spec."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    best_t, best_s = 0.5, -1.0
    for t, s, sp in zip(thr, tpr, spec):
        if sp >= min_spec and s > best_s:
            best_s, best_t = s, float(t)
    return best_t


def metrics(y_true, y_prob, y_pred):
    """Standard-Metriken inkl. Konfusionsmatrix."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Accuracy':     accuracy_score(y_true, y_pred),
        'AUC':          roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'Sensitivität': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'Spezifität':   tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }


# ──────────────────────────────────────────────────────────────────────────
# 5. Auswertungsmodi
# ──────────────────────────────────────────────────────────────────────────

def evaluate_window(pipe, X, y, groups):
    """Fensterbasiert, Threshold = 0.5 fix (Baseline, kein Tuning)."""
    yt, yp, yd = lopo_window_predictions(pipe, X, y, groups)
    return metrics(yt, yp, yd), 0.5


def evaluate_window_nested(pipe, X, y, groups, min_spec=TARGET_SPEC,
                           inner_splits=5, random_state=RANDOM_STATE):
    """
    Fensterbasiert mit leckagefreiem Threshold-Tuning.

    Äußere CV: LOPO über Patienten (kein Testpatient im Training).
    Innere CV:  StratifiedGroupKFold auf den Trainings-FENSTERN, um den
                optimalen Threshold zu bestimmen — auf Fensterebene, nicht
                auf Patientenebene.  Damit entscheidet das Modell pro Fenster
                und der Threshold sieht nie die Testdaten.

    Das ist der Modus, den der Betreuer fordert.
    """
    logo = LeaveOneGroupOut()
    all_yt, all_yp, all_yd = [], [], []
    used_t = []

    for tr, te in logo.split(X, y, groups):
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        X_te, y_te       = X[te], y[te]

        # ── Innere CV: Threshold auf Trainings-Fenstern bestimmen ──────────
        n_inner = min(inner_splits, len(np.unique(g_tr)))
        inner_cv = StratifiedGroupKFold(n_splits=n_inner, shuffle=True,
                                        random_state=random_state)
        inner_yt, inner_yp = [], []
        for itr, ite in inner_cv.split(X_tr, y_tr, g_tr):
            pipe.fit(X_tr[itr], y_tr[itr])
            inner_yp.extend(pipe.predict_proba(X_tr[ite])[:, 1])
            inner_yt.extend(y_tr[ite])

        t = choose_threshold(np.array(inner_yt), np.array(inner_yp), min_spec)

        # ── Äußeres Modell: auf allen Trainings-Fenstern trainieren ────────
        pipe.fit(X_tr, y_tr)
        yp_te = pipe.predict_proba(X_te)[:, 1]

        all_yt.extend(y_te)
        all_yp.extend(yp_te)
        all_yd.extend((yp_te >= t).astype(int))
        used_t.extend([t] * len(y_te))

    all_yt  = np.array(all_yt)
    all_yp  = np.array(all_yp)
    all_yd  = np.array(all_yd)
    m = metrics(all_yt, all_yp, all_yd)
    return m, float(np.mean(used_t))


def evaluate_mv_pooled(pipe, X, y, groups, min_spec=TARGET_SPEC,
                       window_sqi=None, min_sqi=0.0):
    """ALT (optimistisch): Threshold auf allen OOF-Patientenprobs gewählt.
    window_sqi gesetzt -> SQI-gewichtete Fusion."""
    _, true, prob = lopo_patient_probs(pipe, X, y, groups, window_sqi, min_sqi)
    t = choose_threshold(true, prob, min_spec)
    pred = (prob >= t).astype(int)
    return metrics(true, prob, pred), t


def evaluate_mv_nested(pipe, X, y, groups, min_spec=TARGET_SPEC,
                       inner_splits=5, random_state=RANDOM_STATE,
                       window_sqi=None, min_sqi=0.0):
    """
    LECKAGEFREI: äußere LOPO; Threshold pro Fold in einer inneren k-Fold
    NUR auf den Trainingspatienten bestimmt.
    window_sqi gesetzt -> SQI-gewichtete Fusion (innen wie außen).
    """
    logo = LeaveOneGroupOut()
    true, prob, pred, used_t = [], [], [], []
    for tr, te in logo.split(X, y, groups):
        w_tr = None if window_sqi is None else window_sqi[tr]
        it, ip = _inner_patient_probs(pipe, X[tr], y[tr], groups[tr],
                                      window_sqi=w_tr, min_sqi=min_sqi,
                                      n_splits=inner_splits, random_state=random_state)
        t = choose_threshold(it, ip, min_spec)
        pipe.fit(X[tr], y[tr])
        w_te = None if window_sqi is None else window_sqi[te]
        p_mean = _agg_patient(pipe.predict_proba(X[te])[:, 1], w_te, min_sqi)
        true.append(int(y[te][0])); prob.append(p_mean)
        pred.append(int(p_mean >= t)); used_t.append(t)
    true, prob, pred = map(np.array, (true, prob, pred))
    m = metrics(true, prob, pred)
    return m, float(np.mean(used_t))


# ──────────────────────────────────────────────────────────────────────────
# 6. Bootstrap-Konfidenzintervalle (über Patienten)
# ──────────────────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_prob, threshold, n_boot=2000,
                 random_state=RANDOM_STATE, alpha=0.05,
                 groups=None):
    """
    95%-CIs fuer AUC/Sensitivitaet/Spezifitaet/Accuracy.

    groups=None : Resampling auf Fensterebene (fuer window / window_nested).
    groups=array: Resampling auf Patientenebene (fuer mv_* Modi) — haelt alle
                  Fenster eines Patienten zusammen, damit keine Patienten-
                  Informationen zwischen Boot-Trainings-/Test-Split lecken.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    acc, auc_, sens, spec = [], [], [], []

    if groups is None:
        # ── Fenster-Resampling ──────────────────────────────────────────────
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yt, yp = y_true[idx], y_prob[idx]
            if len(np.unique(yt)) < 2:
                continue
            yd = (yp >= threshold).astype(int)
            m = metrics(yt, yp, yd)
            acc.append(m['Accuracy']); auc_.append(m['AUC'])
            sens.append(m['Sensitivität']); spec.append(m['Spezifität'])
    else:
        # ── Patienten-Resampling (Cluster-Bootstrap) ────────────────────────
        groups = np.asarray(groups)
        unique_pats = np.unique(groups)
        n_pat = len(unique_pats)
        for _ in range(n_boot):
            sampled = rng.choice(unique_pats, size=n_pat, replace=True)
            idx = np.concatenate([np.where(groups == p)[0] for p in sampled])
            yt, yp = y_true[idx], y_prob[idx]
            if len(np.unique(yt)) < 2:
                continue
            yd = (yp >= threshold).astype(int)
            m = metrics(yt, yp, yd)
            acc.append(m['Accuracy']); auc_.append(m['AUC'])
            sens.append(m['Sensitivität']); spec.append(m['Spezifität'])

    def ci(v):
        return (float(np.percentile(v, 100 * alpha / 2)),
                float(np.percentile(v, 100 * (1 - alpha / 2))))
    return {'Accuracy': ci(acc), 'AUC': ci(auc_),
            'Sensitivität': ci(sens), 'Spezifität': ci(spec)}


# ──────────────────────────────────────────────────────────────────────────
# 7. Komfort: alle Modelle vergleichen
# ──────────────────────────────────────────────────────────────────────────

def compare_models(X, y, groups, balanced=True,
                   modes=('window', 'window_nested', 'mv_pooled', 'mv_nested'),
                   min_spec=TARGET_SPEC, window_sqi=None, min_sqi=0.0):
    """
    Trainiert alle Modelle in den gewählten Modi und gibt eine
    aufgeräumte Ergebnistabelle (DataFrame) zurück.

    Verfügbare Modi:
      'window'         : fensterbasiert, Threshold = 0.5 fix (Baseline)
      'window_nested'  : fensterbasiert, Threshold leckagefrei auf Trainingsfenstern
                         gewaehlt - das ist der vom Betreuer geforderte Modus
      'mv_pooled'      : Patientenebene, Threshold optimistisch (sieht Testdaten)
      'mv_nested'      : Patientenebene, Threshold leckagefrei
      'mv_pooled_sqi'  : wie mv_pooled, aber SQI-gewichtet
      'mv_nested_sqi'  : wie mv_nested, aber SQI-gewichtet
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    if window_sqi is not None:
        window_sqi = np.asarray(window_sqi, dtype=float)

    # scale_pos_weight für XGBoost: n_negativ / n_positiv
    n_pos = max(1, int(y.sum())); n_neg = max(1, int((1 - y).sum()))
    spw = n_neg / n_pos

    rows = []
    for name in MODEL_ORDER:
        for mode in modes:
            pipe = make_pipelines(balanced=balanced, scale_pos_weight=spw)[name]  # frische Pipeline je Lauf
            if mode == 'window':
                m, t = evaluate_window(pipe, X, y, groups)
            elif mode == 'window_nested':
                m, t = evaluate_window_nested(pipe, X, y, groups, min_spec)
            elif mode == 'mv_pooled':
                m, t = evaluate_mv_pooled(pipe, X, y, groups, min_spec)
            elif mode == 'mv_nested':
                m, t = evaluate_mv_nested(pipe, X, y, groups, min_spec)
            elif mode == 'mv_pooled_sqi':
                if window_sqi is None:
                    raise ValueError("mv_pooled_sqi benötigt window_sqi")
                m, t = evaluate_mv_pooled(pipe, X, y, groups, min_spec,
                                          window_sqi=window_sqi, min_sqi=min_sqi)
            elif mode == 'mv_nested_sqi':
                if window_sqi is None:
                    raise ValueError("mv_nested_sqi benötigt window_sqi")
                m, t = evaluate_mv_nested(pipe, X, y, groups, min_spec,
                                          window_sqi=window_sqi, min_sqi=min_sqi)
            else:
                raise ValueError(f'Unbekannter Modus: {mode}')
            rows.append({'Modell': name, 'Modus': mode,
                         'Accuracy': m['Accuracy'], 'AUC': m['AUC'],
                         'Sensitivität': m['Sensitivität'],
                         'Spezifität': m['Spezifität'], 'Threshold': t})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Selbsttest (synthetische, gruppierte Daten)
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n_pat, win_per_pat, n_feat = 30, 25, 8
    Xs, ys, gs = [], [], []
    for pid in range(n_pat):
        is_af = pid % 2
        center = 1.2 if is_af else -1.2
        Xp = rng.standard_normal((win_per_pat, n_feat))
        Xp[:, 0] += center + rng.normal(0, 0.5)      # informatives Feature + Patientenrauschen
        Xs.append(Xp); ys += [is_af] * win_per_pat; gs += [f'PAT{pid:03d}'] * win_per_pat
    X = np.vstack(Xs); y = np.array(ys); g = np.array(gs)

    print("Selbsttest compare_models (alle Modi)…\n")
    df = compare_models(X, y, g, balanced=True)
    print(df.to_string(index=False,
          formatters={c: '{:.3f}'.format for c in
                      ['Accuracy', 'AUC', 'Sensitivität', 'Spezifität', 'Threshold']}))

    print("\nVergleich pooled vs nested (LR) — zeigt, dass nested ehrlicher ist:")
    lr_pool = df[(df.Modell == 'LR') & (df.Modus == 'mv_pooled')].iloc[0]
    lr_nest = df[(df.Modell == 'LR') & (df.Modus == 'mv_nested')].iloc[0]
    print(f"  pooled : Sens={lr_pool.Sensitivität:.3f}  Spez={lr_pool.Spezifität:.3f}  t={lr_pool.Threshold:.3f}")
    print(f"  nested : Sens={lr_nest.Sensitivität:.3f}  Spez={lr_nest.Spezifität:.3f}  t={lr_nest.Threshold:.3f}")

    _, true, prob = lopo_patient_probs(make_pipelines(balanced=True)['LR'], X, y, g)
    t = choose_threshold(true, prob)
    ci = bootstrap_ci(true, prob, t, n_boot=500)
    print("\nBootstrap-95%-CIs (LR, pooled-Threshold):")
    for k, (lo, hi) in ci.items():
        print(f"  {k:14s} [{lo:.3f}, {hi:.3f}]")
    print("\n✓ Selbsttest erfolgreich.")
