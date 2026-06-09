"""
audit_plots.py — Metrik- & Graphik-Helfer für die Ergebnis-Auswertung (Audit-Track)
====================================================================================
Bachelorarbeit: AF-Detektion in kontaktlosen Signalen · RWTH Aachen

PARALLELISIERT (joblib/loky, analog zur Zelle "## 5"): alle teuren LOPO-Schleifen
laufen über die äußeren Folds parallel (n_jobs=-1, inner_max_num_threads=1).

Funktionen (jede Grafik trackt window + mv parallel; window = primärer Modus):
  1) per_patient_window_metrics + plot_per_patient_accuracy   — Per-Patient-Fenstergenauigkeit
  2) results_table_with_ci                                    — Tabelle Modell×Modus + Cluster-CI + LaTeX
  3) gating_curve(+ _from_oof) + plot_gating_curve            — Coverage-Genauigkeits-Kurve
  4) make_pipelines_with_mlp + plot_model_comparison_bars     — Modellvergleich window vs mv (inkl. MLP)
  5) confusion_window_and_mv + plot_confusion_window_and_mv   — Konfusionsmatrizen window + mv
  6) calibration_window_and_mv + plot_calibration_window_and_mv — Reliability window + mv

Alle rechnenden Funktionen nehmen make_pipe = Callable() -> frische Pipeline UND ein
n_jobs-Argument (Default -1 = alle Kerne). Intern wird EIN Template gebaut und in den
Workern geklont (loky-picklebar — keine Lambda wird verschickt).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from joblib import Parallel, delayed, parallel_config

import oldCode.src.models as M

N_JOBS_DEFAULT = -1


# ══════════════════════════════════════════════════════════════════════════
# Modul-Level-Worker (picklebar für loky) + paralleler LOPO-Kern
# ══════════════════════════════════════════════════════════════════════════

def _wk_window_fixed(tr, te, pipe, X, y, threshold):
    p = clone(pipe).fit(X[tr], y[tr])
    pr = p.predict_proba(X[te])[:, 1]
    return te, y[te].astype(int), pr


def _wk_window_nested(tr, te, pipe, X, y, groups, min_spec, inner_splits, rs):
    Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
    n_in = min(inner_splits, len(np.unique(gtr)))
    cv = StratifiedGroupKFold(n_splits=n_in, shuffle=True, random_state=rs)
    iyt, iyp = [], []
    for itr, ite in cv.split(Xtr, ytr, gtr):
        q = clone(pipe).fit(Xtr[itr], ytr[itr])
        iyp.extend(q.predict_proba(Xtr[ite])[:, 1]); iyt.extend(ytr[ite])
    t = M.choose_threshold(np.array(iyt), np.array(iyp), min_spec)
    p = clone(pipe).fit(Xtr, ytr)
    pr = p.predict_proba(X[te])[:, 1]
    return te, y[te].astype(int), pr, float(t)


def _wk_patient(tr, te, pipe, X, y, groups, mode, min_spec, inner_splits, rs, wsqi, min_sqi):
    if mode == 'mv_pooled':
        t = None
    else:
        wtr = None if wsqi is None else wsqi[tr]
        it, ip = M._inner_patient_probs(clone(pipe), X[tr], y[tr], groups[tr],
                                        window_sqi=wtr, min_sqi=min_sqi,
                                        n_splits=inner_splits, random_state=rs)
        t = float(M.choose_threshold(it, ip, min_spec))
    p = clone(pipe).fit(X[tr], y[tr])
    wte = None if wsqi is None else wsqi[te]
    prob = float(M._agg_patient(p.predict_proba(X[te])[:, 1], wte, min_sqi))
    return groups[te][0], int(y[te][0]), prob, t


def _parallel(jobs, n_jobs):
    with parallel_config(backend='loky', n_jobs=n_jobs, inner_max_num_threads=1):
        return Parallel()(jobs)


def _oof_window_nested(make_pipe, X, y, groups, min_spec=M.TARGET_SPEC,
                       inner_splits=5, random_state=M.RANDOM_STATE, n_jobs=N_JOBS_DEFAULT):
    """Parallele LOPO-OOF (Fensterebene, genesteter Threshold). Arrays in ORIGINAL-Reihenfolge."""
    X = np.asarray(X, float); y = np.asarray(y).astype(int); groups = np.asarray(groups)
    pipe = make_pipe()
    sp = list(LeaveOneGroupOut().split(X, y, groups))
    out = _parallel((delayed(_wk_window_nested)(tr, te, pipe, X, y, groups,
                     min_spec, inner_splits, random_state) for tr, te in sp), n_jobs)
    n = len(y); yp = np.empty(n); yt = np.empty(n, int); thr = np.empty(n)
    for te, yte, pr, t in out:
        yp[te] = pr; yt[te] = yte; thr[te] = t
    yd = (yp >= thr).astype(int)
    return yt, yp, yd, groups.copy(), thr


def _oof_window_fixed(make_pipe, X, y, groups, threshold=0.5, n_jobs=N_JOBS_DEFAULT):
    """Parallele LOPO-OOF (Fensterebene, fester Threshold). Arrays in ORIGINAL-Reihenfolge."""
    X = np.asarray(X, float); y = np.asarray(y).astype(int); groups = np.asarray(groups)
    pipe = make_pipe()
    sp = list(LeaveOneGroupOut().split(X, y, groups))
    out = _parallel((delayed(_wk_window_fixed)(tr, te, pipe, X, y, threshold) for tr, te in sp), n_jobs)
    n = len(y); yp = np.empty(n); yt = np.empty(n, int)
    for te, yte, pr in out:
        yp[te] = pr; yt[te] = yte
    yd = (yp >= threshold).astype(int)
    return yt, yp, yd, groups.copy(), np.full(n, float(threshold))


def _oof_patient(make_pipe, X, y, groups, mode='mv_nested', min_spec=M.TARGET_SPEC,
                 inner_splits=5, random_state=M.RANDOM_STATE,
                 window_sqi=None, min_sqi=0.0, n_jobs=N_JOBS_DEFAULT):
    """Parallele LOPO-OOF (Patientenebene). mode: 'mv_pooled'|'mv_nested'|'mv_nested_sqi'."""
    X = np.asarray(X, float); y = np.asarray(y).astype(int); groups = np.asarray(groups)
    wsqi = None if window_sqi is None else np.asarray(window_sqi, float)
    if mode == 'mv_nested_sqi' and wsqi is None:
        raise ValueError("mv_nested_sqi benötigt window_sqi")
    use_w = wsqi if mode == 'mv_nested_sqi' else None
    pipe = make_pipe()
    sp = list(LeaveOneGroupOut().split(X, y, groups))
    out = _parallel((delayed(_wk_patient)(tr, te, pipe, X, y, groups, mode,
                     min_spec, inner_splits, random_state, use_w, min_sqi) for tr, te in sp), n_jobs)
    ids = np.array([o[0] for o in out])
    true = np.array([o[1] for o in out], int)
    prob = np.array([o[2] for o in out], float)
    ts = [o[3] for o in out]
    if mode == 'mv_pooled':
        t = M.choose_threshold(true, prob, min_spec)
    else:
        t = float(np.mean([x for x in ts if x is not None]))
    pred = (prob >= t).astype(int)
    return true, prob, pred, ids, float(t)


# ══════════════════════════════════════════════════════════════════════════
# 1) Per-Patient-Fenstergenauigkeit
# ══════════════════════════════════════════════════════════════════════════

def per_patient_window_metrics(make_pipe, X, y, groups, min_spec=M.TARGET_SPEC,
                               nested=True, threshold=0.5, n_jobs=N_JOBS_DEFAULT):
    """
    Per-Patient-Fenstergenauigkeit auf LOPO-OOF-Fenstern (parallel).
    nested=True -> genesteter Threshold (konsistent zu window_nested).
    Bei persistentem AF: AF-Patient -> accuracy == Fenster-Sensitivität,
    Non-AF-Patient -> accuracy == Fenster-Spezifität.
    """
    if nested:
        yt, yp, yd, pat, _ = _oof_window_nested(make_pipe, X, y, groups, min_spec, n_jobs=n_jobs)
    else:
        yt, yp, yd, pat, _ = _oof_window_fixed(make_pipe, X, y, groups, threshold, n_jobs=n_jobs)
    rows = []
    for p in pd.unique(pat):
        m = pat == p
        rows.append({'patient': p, 'AF': int(yt[m][0]), 'n_windows': int(m.sum()),
                     'accuracy': float(accuracy_score(yt[m], yd[m])),
                     'median_prob': float(np.median(yp[m]))})
    return pd.DataFrame(rows).sort_values(['AF', 'accuracy']).reset_index(drop=True)


def plot_per_patient_accuracy(df_pp, save_path=None, title=None):
    """Boxplot (AF vs Non-AF) der Per-Patient-Fenstergenauigkeit + Einzelpunkte (Jitter)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    data, labels, colors = [], [], []
    for lbl, name, col in [(0, 'Non-AF', '#4C72B0'), (1, 'AF', '#C44E52')]:
        vals = df_pp.loc[df_pp.AF == lbl, 'accuracy'].values
        if len(vals):
            data.append(vals); labels.append(f'{name}\n(n={len(vals)})'); colors.append(col)
    bp = ax.boxplot(data, labels=labels, widths=0.5, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=6))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    for med in bp['medians']:
        med.set_color('black'); med.set_linewidth(1.5)
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        ax.scatter(rng.normal(i + 1, 0.06, size=len(vals)), vals, s=34,
                   color=colors[i], edgecolor='white', zorder=3, linewidth=0.6)
    ax.set_ylabel('Fenstergenauigkeit je Patient'); ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color='gray', ls='--', lw=1, label='Zufall (0.5)')
    ax.set_title(title or 'Per-Patient-Fenstergenauigkeit (LOPO)')
    ax.legend(loc='lower left', fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════
# 2) Konsolidierte Ergebnistabelle mit Cluster-Bootstrap-CI
# ══════════════════════════════════════════════════════════════════════════

def _ci_str(lo, hi):
    return f'[{lo:.3f}, {hi:.3f}]'


def results_table_with_ci(make_pipelines_fn, X, y, groups,
                          models=None, modes=('window_nested', 'mv_nested'),
                          min_spec=M.TARGET_SPEC, window_sqi=None, min_sqi=0.0,
                          n_boot=2000, balanced=True, random_state=M.RANDOM_STATE,
                          scale_pos_weight=1.0, n_jobs=N_JOBS_DEFAULT):
    """
    Punktmetriken + 95%-Cluster-Bootstrap-CI je (Modell, Modus). Parallel über LOPO-Folds.
    window-Modi -> Cluster-Bootstrap (Patienten je Fenster); mv-Modi -> Patienten-Resampling.
    make_pipelines_fn(balanced, scale_pos_weight) -> dict{name: pipeline}.
    Rückgabe-DataFrame; .attrs['latex'] = LaTeX-tabular.
    """
    models = models or M.MODEL_ORDER
    X = np.asarray(X, float); y = np.asarray(y).astype(int); groups = np.asarray(groups)
    wsqi = None if window_sqi is None else np.asarray(window_sqi, float)

    def mp(name):
        return make_pipelines_fn(balanced=balanced, scale_pos_weight=scale_pos_weight)[name]

    rows = []
    for name in models:
        for mode in modes:
            if mode == 'window':
                yt, yp, yd, pat, _ = _oof_window_fixed(lambda: mp(name), X, y, groups, 0.5, n_jobs=n_jobs)
                boot_groups, thr = pat, 0.5
            elif mode == 'window_nested':
                yt, yp, yd, pat, thrarr = _oof_window_nested(lambda: mp(name), X, y, groups, min_spec, n_jobs=n_jobs)
                boot_groups, thr = pat, float(np.mean(thrarr))
            elif mode in ('mv_pooled', 'mv_nested', 'mv_nested_sqi'):
                yt, yp, yd, ids, thr = _oof_patient(lambda: mp(name), X, y, groups, mode=mode,
                                                    min_spec=min_spec, window_sqi=wsqi,
                                                    min_sqi=min_sqi, n_jobs=n_jobs)
                boot_groups = None
            else:
                raise ValueError(f'Unbekannter Modus: {mode}')
            m = M.metrics(yt, yp, yd)
            ci = M.bootstrap_ci(yt, yp, thr, n_boot=n_boot, groups=boot_groups, random_state=random_state)
            rows.append({'Modell': name, 'Modus': mode, 'Threshold': round(thr, 3),
                         'AUC': round(m['AUC'], 3),           'AUC_CI': _ci_str(*ci['AUC']),
                         'Sens': round(m['Sensitivität'], 3), 'Sens_CI': _ci_str(*ci['Sensitivität']),
                         'Spez': round(m['Spezifität'], 3),   'Spez_CI': _ci_str(*ci['Spezifität']),
                         'Acc': round(m['Accuracy'], 3),      'Acc_CI': _ci_str(*ci['Accuracy'])})
    df = pd.DataFrame(rows)
    latex_df = pd.DataFrame({
        'Modell': df['Modell'], 'Modus': df['Modus'],
        'AUC':  df.apply(lambda r: f"{r['AUC']:.3f} {r['AUC_CI']}", axis=1),
        'Sens.': df.apply(lambda r: f"{r['Sens']:.3f} {r['Sens_CI']}", axis=1),
        'Spez.': df.apply(lambda r: f"{r['Spez']:.3f} {r['Spez_CI']}", axis=1)})
    df.attrs['latex'] = latex_df.to_latex(index=False, escape=False, column_format='llccc')
    return df


# ══════════════════════════════════════════════════════════════════════════
# 3) Coverage-Genauigkeits-(Gating-)Kurve
# ══════════════════════════════════════════════════════════════════════════

def gating_curve(y_true, y_prob, sqi, n_points=11, q_max=0.85):
    """AUC (& Accuracy) als Funktion der Coverage, wenn nur Fenster mit SQI >= Schwelle bleiben."""
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob, float); sqi = np.asarray(sqi, float)
    yd = (y_prob >= 0.5).astype(int)
    rows = []
    for q in np.linspace(0.0, q_max, n_points):
        t = float(np.quantile(sqi, q)); m = sqi >= t
        if m.sum() < 10 or len(np.unique(y_true[m])) < 2:
            continue
        rows.append({'sqi_threshold': t, 'coverage': float(m.mean()), 'n': int(m.sum()),
                     'AUC': float(roc_auc_score(y_true[m], y_prob[m])),
                     'accuracy': float(accuracy_score(y_true[m], yd[m]))})
    return pd.DataFrame(rows)


def plot_gating_curve(curve_df, save_path=None, title=None, show_accuracy=True):
    """Eigenständige Gating-Abbildung: AUC (+ optional Accuracy) über die Coverage."""
    fig, ax = plt.subplots(figsize=(7, 5))
    cov = curve_df['coverage'].values * 100
    ax.plot(cov, curve_df['AUC'].values, 'o-', color='#C44E52', lw=2, label='AUC')
    if show_accuracy and 'accuracy' in curve_df:
        ax.plot(cov, curve_df['accuracy'].values, 's--', color='#4C72B0', lw=1.8, label='Accuracy (t=0.5)')
    ax.set_xlabel('Coverage [%] (Anteil behaltener Fenster)'); ax.set_ylabel('Fenster-Metrik')
    ax.invert_xaxis()
    ax.set_title(title or 'Coverage-Genauigkeits-Kurve (SQI-Gating)')
    ax.grid(alpha=0.3); ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


def gating_curve_from_oof(make_pipe, X, y, groups, window_sqi, n_jobs=N_JOBS_DEFAULT, **kw):
    """LOPO-OOF (parallel, fester Threshold) -> Gating-Kurve. SQI ist über Original-Index ausgerichtet."""
    yt, yp, _, _, _ = _oof_window_fixed(make_pipe, X, y, groups, 0.5, n_jobs=n_jobs)
    return gating_curve(yt, yp, np.asarray(window_sqi, float), **kw)


# ══════════════════════════════════════════════════════════════════════════
# 4) MLP-Pipeline + Modellvergleich-Balken (window_nested vs mv_nested)
# ══════════════════════════════════════════════════════════════════════════

def make_pipelines_with_mlp(balanced=True, random_state=M.RANDOM_STATE, scale_pos_weight=1.0):
    """Wie M.make_pipelines, plus ein kleines MLP als EIN bestätigender Vergleichsbalken."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    pipes = M.make_pipelines(balanced=balanced, random_state=random_state, scale_pos_weight=scale_pos_weight)
    pipes['MLP'] = Pipeline([
        M._clean_step(),
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, activation='relu',
                              solver='adam', max_iter=400, early_stopping=True,
                              n_iter_no_change=15, random_state=random_state))])
    return pipes


MODEL_ORDER_MLP = M.MODEL_ORDER + ['MLP']


def plot_model_comparison_bars(ci_table, metric='AUC', modes=('window_nested', 'mv_nested'),
                               save_path=None, title=None, mode_labels=None, mode_colors=None):
    """
    Gruppierte Balken je Modell: window_nested vs mv_nested für EINE Metrik.
    CI-Fehlerbalken OPTIONAL (funktioniert auch auf res aus Zelle "## 5", ohne CI-Spalten).
    """
    mode_labels = mode_labels or {'window_nested': 'window (primär)', 'mv_nested': 'mv (Decke)',
                                  'window': 'window t=0.5', 'mv_nested_sqi': 'mv+SQI', 'mv_pooled': 'mv pooled'}
    mode_colors = mode_colors or {'window_nested': '#C44E52', 'mv_nested': '#4C72B0',
                                  'window': '#DD8452', 'mv_nested_sqi': '#55A868', 'mv_pooled': '#8172B3'}
    ci_col = f'{metric}_CI'; has_ci = ci_col in ci_table.columns
    models = list(dict.fromkeys(ci_table['Modell']))
    n_modes = len(modes); width = 0.8 / n_modes; x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(models)), 5))
    for j, mode in enumerate(modes):
        vals, los, his = [], [], []
        for mdl in models:
            row = ci_table[(ci_table.Modell == mdl) & (ci_table.Modus == mode)]
            if row.empty:
                vals.append(np.nan); los.append(0); his.append(0); continue
            v = float(row[metric].iloc[0]); vals.append(v)
            if has_ci:
                lo, hi = row[ci_col].iloc[0].strip('[]').split(',')
                los.append(v - float(lo)); his.append(float(hi) - v)
            else:
                los.append(0); his.append(0)
        ax.bar(x + j * width - 0.4 + width / 2, vals, width,
               yerr=([los, his] if has_ci else None), capsize=3,
               label=mode_labels.get(mode, mode), color=mode_colors.get(mode, None),
               edgecolor='white', error_kw=dict(lw=1, alpha=0.7))
        for xi, v in zip(x + j * width - 0.4 + width / 2, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.006, f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel(metric)
    lo_lim = min(0.7, float(np.nanmin(ci_table[metric].values)) - 0.05)
    ax.set_ylim(max(0.0, lo_lim), 1.02)
    ax.set_title(title or f'Modellvergleich {metric}: Fenster- vs. Patientenebene')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════
# 5) Konfusionsmatrizen an Betriebspunkten (window_nested + mv_nested)
# ══════════════════════════════════════════════════════════════════════════

def confusion_window_and_mv(make_pipe, X, y, groups, min_spec=M.TARGET_SPEC,
                            window_sqi=None, min_sqi=0.0, mv_mode='mv_nested', n_jobs=N_JOBS_DEFAULT):
    """Beide Konfusionsmatrizen EINES Modells: window_nested (Fenster) + mv_nested (Patienten). Parallel."""
    yt_w, yp_w, yd_w, _, thr_w = _oof_window_nested(make_pipe, X, y, groups, min_spec, n_jobs=n_jobs)
    cm_w = confusion_matrix(yt_w, yd_w, labels=[0, 1])
    yt_m, yp_m, yd_m, _, thr_m = _oof_patient(make_pipe, X, y, groups, mode=mv_mode,
                                              min_spec=min_spec, window_sqi=window_sqi,
                                              min_sqi=min_sqi, n_jobs=n_jobs)
    cm_m = confusion_matrix(yt_m, yd_m, labels=[0, 1])
    return {'window': (cm_w, len(yt_w)), 'mv': (cm_m, len(yt_m)),
            'thr_window': float(np.mean(thr_w)), 'thr_mv': float(thr_m)}


def plot_confusion_window_and_mv(cm_dict, save_path=None, model_name=''):
    """Zwei Konfusionsmatrizen nebeneinander: Fensterebene (primär) links, Patientenebene rechts."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('c', ['#f7fbff', '#C44E52'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panels = [('window', f'Fensterebene · window_nested\n(t={cm_dict["thr_window"]:.3f}, n={cm_dict["window"][1]})'),
              ('mv',     f'Patientenebene · mv_nested\n(t={cm_dict["thr_mv"]:.3f}, n={cm_dict["mv"][1]})')]
    for ax, (key, ttl) in zip(axes, panels):
        cm, _ = cm_dict[key]
        ax.imshow(cm, cmap=cmap, aspect='equal')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold',
                        color='white' if cm[i, j] > cm.max() * 0.55 else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Non-AF', 'AF'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Non-AF', 'AF'])
        ax.set_xlabel('Vorhersage'); ax.set_ylabel('Wahrheit'); ax.set_title(ttl, fontsize=10)
    fig.suptitle(f'Konfusionsmatrizen {model_name}'.strip(), y=1.02, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, axes


# ══════════════════════════════════════════════════════════════════════════
# 6) Kalibrierung / Reliability (window_nested + mv_nested)
# ══════════════════════════════════════════════════════════════════════════

def calibration_window_and_mv(make_pipe, X, y, groups, min_spec=M.TARGET_SPEC,
                              window_sqi=None, min_sqi=0.0, mv_mode='mv_nested',
                              n_bins_window=10, n_bins_mv=5, n_jobs=N_JOBS_DEFAULT):
    """Kalibrierungsdaten + Brier-Score für beide Ebenen. Parallel."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    yt_w, yp_w, _, _, _ = _oof_window_nested(make_pipe, X, y, groups, min_spec, n_jobs=n_jobs)
    fp_w, mp_w = calibration_curve(yt_w, yp_w, n_bins=n_bins_window, strategy='quantile')
    yt_m, yp_m, _, _, _ = _oof_patient(make_pipe, X, y, groups, mode=mv_mode, min_spec=min_spec,
                                       window_sqi=window_sqi, min_sqi=min_sqi, n_jobs=n_jobs)
    fp_m, mp_m = calibration_curve(yt_m, yp_m, n_bins=n_bins_mv, strategy='quantile')
    return {'window': {'frac_pos': fp_w, 'mean_pred': mp_w,
                       'brier': float(brier_score_loss(yt_w, yp_w)), 'n': len(yt_w)},
            'mv':     {'frac_pos': fp_m, 'mean_pred': mp_m,
                       'brier': float(brier_score_loss(yt_m, yp_m)), 'n': len(yt_m)}}


def plot_calibration_window_and_mv(cal_dict, save_path=None, title=None):
    """Reliability-Diagramm: window (primär) + mv auf einer Achse, mit Brier-Score in der Legende."""
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 1], [0, 1], ls='--', color='gray', lw=1, label='perfekt kalibriert')
    w, m = cal_dict['window'], cal_dict['mv']
    ax.plot(w['mean_pred'], w['frac_pos'], 'o-', color='#C44E52', lw=2,
            label=f"window_nested (Brier={w['brier']:.3f}, n={w['n']})")
    ax.plot(m['mean_pred'], m['frac_pos'], 's--', color='#4C72B0', lw=2,
            label=f"mv_nested (Brier={m['brier']:.3f}, n={m['n']})")
    ax.set_xlabel('mittlere vorhergesagte Wahrscheinlichkeit'); ax.set_ylabel('beobachteter AF-Anteil')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or 'Kalibrierung: Fenster- vs. Patientenebene')
    ax.legend(loc='upper left', fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════
# 7) SQI-Aggregations-Robustheit (beantwortet: sind die Composite-Gewichte egal?)
# ══════════════════════════════════════════════════════════════════════════
# Voraussetzung: die Extraktion speichert die Einzel-SQIs als Metadaten-Spalten
#   '_sqi_{kanal}_{index}'  (index in kSQI,sSQI,pSQI,bSQI,tSQI,composite).
# Diese Spalten sind NICHT im ML-Feature-Satz (Präfix '_' -> beim X-Bau gedroppt).

def _agg_channel_sqi(sqi_df, index, signals, agg='mean'):
    """Aggregiert einen Einzel-SQI über die Kanäle zu einem Fenster-Skalar."""
    cols = [f'_sqi_{s}_{index}' for s in signals if f'_sqi_{s}_{index}' in sqi_df.columns]
    if not cols:
        raise KeyError(f"Keine '_sqi_*_{index}'-Spalten gefunden — Extraktion mit SQI_DETAIL=True neu laufen lassen.")
    Mx = sqi_df[cols].to_numpy(float)
    if agg == 'mean':   return np.nanmean(Mx, axis=1)
    if agg == 'median': return np.nanmedian(Mx, axis=1)
    if agg == 'min':    return np.nanmin(Mx, axis=1)
    raise ValueError(agg)


def make_default_gating_schemes(signals):
    def comp(df): return _agg_channel_sqi(df, 'composite', signals)
    def ps(df):   return _agg_channel_sqi(df, 'pSQI', signals)
    def ts(df):   return _agg_channel_sqi(df, 'tSQI', signals)
    def bs(df):   return _agg_channel_sqi(df, 'bSQI', signals)          # <- NEU
    def eq(df):   return 0.33*_agg_channel_sqi(df,'tSQI',signals) + 0.33*_agg_channel_sqi(df,'pSQI',signals) + 0.33*_agg_channel_sqi(df, "bSQI", signals)
    return {'composite (0.5/0.3/0.2)': comp, 'pSQI allein (Li 2008)': ps,
            'tSQI allein (Orphanidou 2015)': ts, 'bSQI allein (Behar 2013)': bs,   # <- NEU
            'Gleichgewicht t+p+b': eq}


def gating_robustness(make_pipe, X, y, groups, sqi_df, signals,
                      schemes=None, n_points=9, n_jobs=N_JOBS_DEFAULT):
    """
    Berechnet die Gating-Kurve unter mehreren SQI-Aggregationen auf DENSELBEN OOF-Probs.
    Liegen die Kurven übereinander, sind die Composite-Gewichte nicht tragend.
    sqi_df muss zeilengleich/ordnungsgleich zu X sein (z.B. die volle df).
    Rückgabe: dict{scheme_name: gating_curve-DataFrame}.
    """
    yt, yp, _, _, _ = _oof_window_fixed(make_pipe, X, y, groups, 0.5, n_jobs=n_jobs)  # OOF EINMAL
    schemes = schemes or make_default_gating_schemes(signals)
    return {name: gating_curve(yt, yp, np.asarray(fn(sqi_df), float), n_points=n_points)
            for name, fn in schemes.items()}


def plot_gating_robustness(curves, save_path=None, title=None):
    """Überlagert die Gating-Kurven (AUC über Coverage) aller Aggregationsschemata."""
    styles = ['o-', 's--', '^-.', 'd:', 'v-']
    colors = ['#C44E52', '#4C72B0', '#55A868', '#8172B3', '#DD8452']
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (name, cur), st, col in zip(curves.items(), styles, colors):
        ax.plot(cur['coverage'].values * 100, cur['AUC'].values, st, color=col, lw=1.8, label=name)
    ax.invert_xaxis()
    ax.set_xlabel('Coverage [%] (Anteil behaltener Fenster)'); ax.set_ylabel('AUC (Fensterebene)')
    ax.set_title(title or 'Robustheit des Gatings gegenüber der SQI-Aggregation')
    ax.grid(alpha=0.3); ax.legend(loc='lower left', fontsize=8.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig, ax
