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

import os
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

import experts as X
import gating as G
import models as M
import extract as E

# Verzeichnis dieser Datei (src/) — robuste Pfadangabe für joblib/loky-Worker,
# unabhängig vom Arbeitsverzeichnis des Notebooks.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))


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
                 gate_params: dict | None = None,
                 weight_map: str = 'softmax', map_params: dict | None = None,
                 freeze_weights: bool = False,
                 return_arrays: bool = False, return_weights: bool = False,
                 fold_cache: dict | None = None):
    """
    Fensterweise AF-Auswertung der Mixture of Experts in patientenweiser LOPO-CV.

    gate_kind : 'mlp' | 'gb' | 'ridge'  -> gelerntes Gate (B)
                'equal'                  -> Gleichgewichts-Baseline (naive Fusion)

    freeze_weights : nur für gelernte Gates. Das Gate wird wie gewohnt trainiert,
        aber statt der FENSTERWEISEN Gewichte wird je Fold das mittlere
        Trainingsgewicht (Spaltenmittel) eingefroren und KONSTANT auf alle
        Test-Fenster angewendet. Das isoliert exakt den Beitrag der
        fensterweisen SQI-Adaptivität: gefroren vs. gelernt = was die
        Pro-Fenster-Anpassung tatsächlich bringt. (Bei 'equal' wirkungslos.)

    return_weights : hängt zusätzlich die fensterweise Gewichtsmatrix yw (n, 3)
        in Reihenfolge G.ORDER an die Rückgabe an (für die Adaptivitäts-Diagnose).

    fold_cache : OPTIONAL. Ergebnis von `precompute_folds(...)`. Enthält je äußerem
        Fold die bereits berechnete (TEURE) Experten-Schicht — OOF-Trainings-Probs
        `P_tr` und Test-Probs `P_te`. Da diese NUR von (Daten, clf_per_modality,
        inner_splits, random_state, Fold-Split) abhängen und NICHT von
        gate_kind/target_metric/freeze_weights, werden sie einmal berechnet und über
        alle Gate-Varianten wiederverwendet (Faktor ~10 weniger Experten-Fits).
        LECKAGE-SICHERUNG: der Cache trägt einen Hash seiner Eingänge; passt er nicht
        EXAKT zu (df, y, groups, clf_per_modality, inner_splits, random_state), wird
        er VERWORFEN (ValueError) statt stillschweigend falsch verwendet.

    Rückgabe : (metrics_dict, mean_threshold
                [, y_true, y_prob, y_pred, y_groups]      falls return_arrays
                [, y_weights]                              falls return_weights)
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    rel_al = align_reliability(df, rel)
    gate_cols = E.gate_sqi_cols(df, 'all')

    # Quelle der (tr, te, P_tr, P_te) je Fold: entweder live berechnet (unveränderte
    # Logik) ODER aus dem geprüften Cache. Der NACHGELAGERTE Code ist in beiden
    # Fällen IDENTISCH -> Cache ändert das Ergebnis nicht, nur die Laufzeit.
    if fold_cache is not None:
        _verify_fold_cache(fold_cache, df, y, groups, clf_per_modality,
                           inner_splits, random_state)
        fold_iter = [(f['tr'], f['te'], f['P_tr'], f['P_te'])
                     for f in fold_cache['folds']]
    else:
        logo = LeaveOneGroupOut()
        fold_iter = []
        for tr, te in logo.split(df, y, groups):
            # 1) leckagefreie OOF-Experten-Probs auf den Trainingsfenstern
            oof_tr = X.oof_expert_probs(df.iloc[tr], y[tr], groups[tr], clf_per_modality,
                                        n_splits=inner_splits, random_state=random_state)
            P_tr = G.probs_matrix(oof_tr)
            # 5a) Experten auf ALLEN Trainingspatienten fitten, auf Test anwenden
            experts = X.fit_experts(df.iloc[tr], y[tr], clf_per_modality,
                                    random_state=random_state)
            P_te = G.probs_matrix(X.expert_probs(experts, df.iloc[te]))
            fold_iter.append((np.asarray(tr), np.asarray(te), P_tr, P_te))

    yt, yp, yd, yg, used_t, wt = [], [], [], [], [], []
    for tr, te, P_tr, P_te in fold_iter:
        y_tr = y[tr]

        # 3/4) Gate fitten + Trainingsgewichte + Schwelle  (billig, varianten-abhängig)
        w_const = None
        if gate_kind == 'equal':
            w_tr = G.equal_weights(len(tr))
            gate, scale = None, None
        else:
            T_tr = _prep_targets(rel_al.iloc[tr], target_metric)
            gate = G.make_gate(kind=gate_kind, random_state=random_state,
                               **(gate_params or {}))
            gate.fit(df.iloc[tr][gate_cols].values, T_tr)
            err_tr = gate.predict(df.iloc[tr][gate_cols].values)

            # Fehler -> Gewicht: Abbildungs-Parameter NUR aus dem Training ableiten,
            # dann IDENTISCH auf Train und Test anwenden (leckagefrei). 'softmax' =
            # bisheriges Verhalten (Default, unverändert); 'exp' = Bachelet e0/τ (3.6.2).
            if weight_map == 'softmax':
                scale = float(np.nanmedian(err_tr)) + 1e-6   # feste Temperatur aus Training
                _map = lambda e: G.errors_to_weights(e, scale=scale)
            elif weight_map == 'exp':
                mp = map_params or {}
                e0 = float(mp.get('e0', np.nanpercentile(err_tr, 10)))
                tau = float(mp.get('tau', np.nanmedian(err_tr) + 1e-6))
                _map = lambda e: G.errors_to_weights_exp(e, e0=e0, tau=tau)
            else:
                raise ValueError("weight_map muss 'softmax' oder 'exp' sein")

            w_tr = _map(err_tr)
            if freeze_weights:
                # mittleres Trainingsgewicht je Modalität einfrieren (keine Pro-Fenster-Variation)
                w_const = w_tr.mean(axis=0, keepdims=True)
                w_tr = np.repeat(w_const, len(tr), axis=0)

        fused_tr = G.fuse(w_tr, P_tr)
        t = M.choose_threshold(y_tr, fused_tr, min_spec)

        # 5b) Testgewichte bestimmen (KEIN GT im Test!) und fusionieren
        if gate_kind == 'equal':
            w_te = G.equal_weights(len(te))
        elif freeze_weights:
            w_te = np.repeat(w_const, len(te), axis=0)   # eingefrorenes Trainingsmittel
        else:
            w_te = _map(gate.predict(df.iloc[te][gate_cols].values))
        fused_te = G.fuse(w_te, P_te)

        yt.extend(y[te]); yp.extend(fused_te); yd.extend((fused_te >= t).astype(int))
        yg.extend(groups[te]); used_t.append(t); wt.extend(w_te)

    yt, yp, yd = map(np.array, (yt, yp, yd))
    yg = np.array(yg)
    yw = np.asarray(wt, dtype=float)
    m = M.metrics(yt, yp, yd)
    out = (m, float(np.mean(used_t)))
    if return_arrays:
        out = out + (yt, yp, yd, yg)
    if return_weights:
        out = out + (yw,)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Experten-Schicht cachen: einmal je Fold rechnen, über alle Gate-Varianten nutzen
# ──────────────────────────────────────────────────────────────────────────

def _expert_layer_signature(df: pd.DataFrame, y, groups, clf_per_modality,
                            inner_splits: int, random_state: int) -> str:
    """
    Eindeutiger Hash GENAU der Eingänge, die P_tr/P_te bestimmen — also
    Experten-Merkmale (Inhalt + Spaltennamen), Labels, Patienten-Gruppen,
    Klassifikatorwahl, inner_splits, random_state. NICHT enthalten:
    gate_kind, target_metric, freeze_weights, min_spec (verändern P_tr/P_te nicht).

    Zweck = Leckage-/Korrektheits-Sicherung: ein Cache darf nur dann
    wiederverwendet werden, wenn diese Eingänge BIT-genau übereinstimmen.
    """
    clf = clf_per_modality or X.DEFAULT_CLF
    feat_cols = sorted({c for m in clf for c in E.expert_feature_cols(df, m)})
    h = hashlib.sha1()
    h.update('|'.join(feat_cols).encode())
    h.update(np.ascontiguousarray(df[feat_cols].values, dtype=np.float64).tobytes())
    h.update(np.asarray(y).astype(np.int64).tobytes())
    h.update('|'.join(map(str, np.asarray(groups))).encode())
    h.update(repr(sorted(clf.items())).encode())
    h.update(f'{int(inner_splits)}|{int(random_state)}'.encode())
    return h.hexdigest()[:16]


def _fold_expert_layer(args):
    """Worker (loky): rechnet die TEURE Experten-Schicht für EINEN Fold.
    Gibt OOF-Trainings-Probs (n_tr, 3) und Test-Probs (n_te, 3) in G.ORDER zurück.
    Identisch zur Live-Logik in evaluate_moe — nur ausgelagert & parallelisierbar."""
    import sys
    for p in [_SRC_DIR, 'src', '.', '../src']:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'features.py')):
            if p not in sys.path:
                sys.path.insert(0, p)
            break
    import experts as _X
    import gating as _G

    df_tr, y_tr, g_tr, df_te, clf_per_modality, inner_splits, random_state = args
    oof_tr = _X.oof_expert_probs(df_tr, y_tr, g_tr, clf_per_modality,
                                 n_splits=inner_splits, random_state=random_state)
    P_tr = _G.probs_matrix(oof_tr)
    experts = _X.fit_experts(df_tr, y_tr, clf_per_modality, random_state=random_state)
    P_te = _G.probs_matrix(_X.expert_probs(experts, df_te))
    return np.asarray(P_tr, dtype=float), np.asarray(P_te, dtype=float)


def precompute_folds(df: pd.DataFrame, y, groups, clf_per_modality: dict | None = None,
                     inner_splits: int = 5, random_state: int = 42,
                     n_jobs: int = -1, cache_dir: str | None = None,
                     force: bool = False, verbose: bool = True) -> dict:
    """
    Berechnet die Experten-Schicht je LOPO-Fold EINMAL (parallel) und gibt einen
    wiederverwendbaren Cache zurück. Anschließend laufen alle Gate-Varianten
    (compare_gates, compare_adaptivity, gate_weight_report, evaluate_moe) über
    denselben Cache — die teuren Experten-Fits entfallen dort komplett.

    Leckagefrei: pro Fold wird die Experten-Schicht NUR aus den Trainingspatienten
    dieses Folds gebildet (exakt wie bisher in evaluate_moe); es gibt KEIN globales
    OOF, das einen Testpatienten einbeziehen könnte.

    Parallelisierung: loky über die Folds, inner_max_num_threads=1 (kein
    Thread-Oversubscribing der sklearn-Schätzer).

    cache_dir : wenn gesetzt, wird der Cache als
        f'fold_expert_cache_<sig>.joblib' gespeichert/geladen (Re-Runs sind dann
        sofort). force=True erzwingt Neuberechnung.

    Rückgabe (dict):
        sig, clf, inner_splits, random_state,
        folds = [ {tr, te, P_tr, P_te}, ... ]   (tr/te = Integer-Indizes in df)
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    sig = _expert_layer_signature(df, y, groups, clf_per_modality, inner_splits, random_state)

    path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f'fold_expert_cache_{sig}.joblib')
        if os.path.exists(path) and not force:
            import joblib
            if verbose:
                print(f'Fold-Cache gefunden -> lade {path}')
            return joblib.load(path)

    logo = LeaveOneGroupOut()
    folds = list(logo.split(df, y, groups))
    if verbose:
        print(f'Berechne Experten-Schicht für {len(folds)} Folds '
              f'(sig={sig}, n_jobs={n_jobs}) ...')

    tasks = [(df.iloc[tr], y[tr], groups[tr], df.iloc[te],
              clf_per_modality, inner_splits, random_state) for tr, te in folds]

    import time
    from joblib import Parallel, delayed, parallel_config
    t0 = time.time()
    with parallel_config(backend='loky', n_jobs=n_jobs, inner_max_num_threads=1):
        results = Parallel()(delayed(_fold_expert_layer)(a) for a in tasks)

    fold_list = [{'tr': np.asarray(tr), 'te': np.asarray(te), 'P_tr': P_tr, 'P_te': P_te}
                 for (tr, te), (P_tr, P_te) in zip(folds, results)]
    cache = {'sig': sig, 'clf': clf_per_modality or X.DEFAULT_CLF,
             'inner_splits': inner_splits, 'random_state': random_state,
             'folds': fold_list}

    if verbose:
        print(f'  fertig in {time.time()-t0:.1f}s · {len(fold_list)} Folds')
    if path is not None:
        import joblib
        joblib.dump(cache, path)
        if verbose:
            print(f'  gespeichert: {path}')
    return cache


def _verify_fold_cache(fold_cache: dict, df: pd.DataFrame, y, groups,
                       clf_per_modality, inner_splits: int, random_state: int):
    """Leckage-/Korrektheits-Sicherung: der Cache darf NUR bei bit-genauer
    Übereinstimmung der Eingänge verwendet werden, sonst harter Abbruch."""
    sig = _expert_layer_signature(df, np.asarray(y), np.asarray(groups),
                                  clf_per_modality, inner_splits, random_state)
    if fold_cache.get('sig') != sig:
        raise ValueError(
            "fold_cache passt NICHT zu (df, y, groups, clf_per_modality, inner_splits, "
            "random_state) — er wurde mit anderen Eingängen gebaut. Zur Leckage-/"
            "Korrektheits-Sicherung wird er NICHT verwendet. precompute_folds(...) mit "
            "den AKTUELLEN Argumenten neu ausführen (ggf. force=True).")


# ──────────────────────────────────────────────────────────────────────────


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
# Adaptivitäts-Diagnose: bringt die FENSTERWEISE SQI-Steuerung überhaupt etwas?
# ──────────────────────────────────────────────────────────────────────────

def compare_adaptivity(df: pd.DataFrame, rel: pd.DataFrame, y, groups,
                       base_gate: str = 'gb', clf_per_modality: dict | None = None,
                       **kw) -> pd.DataFrame:
    """
    Kernprüfung der Track-B-Behauptung ("SQI steuert die Fusion PRO FENSTER").

    Drei Varianten mit IDENTISCHEN Experten/Schwellen-Routinen, nur die Gewichte
    unterscheiden sich:

        equal      : feste 1/3-Gewichte (naive Fusion).
        <base>-fix : dasselbe gelernte Gate, aber je Fold auf sein mittleres
                     Trainingsgewicht EINGEFROREN — also feste, datengetriebene
                     Modalitätsgewichte OHNE Pro-Fenster-Variation.
        <base>-win : das volle gelernte Gate mit fensterweisen Gewichten.

    Lesart der Tabelle:
        Δ(equal → fix)  = Nutzen einer festen, gelernten Modalitäts-Umgewichtung.
        Δ(fix → win)    = Nutzen der zusätzlichen FENSTERWEISEN SQI-Anpassung.
    Ist Δ(fix → win) ~ 0, trägt die Pro-Fenster-Steuerung nichts bei und das
    Ergebnis ist im Kern eine feste Gewichtung — das gehört dann ehrlich so in
    die Diskussion (zusammen mit dem gate_weight_report, der dasselbe direkt an
    der Gewichtsstreuung zeigt).
    """
    specs = [
        ('equal',                 dict(gate_kind='equal')),
        (f'{base_gate}-fix',      dict(gate_kind=base_gate, freeze_weights=True)),
        (f'{base_gate}-win',      dict(gate_kind=base_gate, freeze_weights=False)),
    ]
    rows = []
    for name, opt in specs:
        m, t = evaluate_moe(df, rel, y, groups, clf_per_modality=clf_per_modality,
                            **opt, **kw)
        rows.append({'variant': name, **m, 'threshold': round(t, 3)})
    return pd.DataFrame(rows)


def gate_weight_report(df: pd.DataFrame, rel: pd.DataFrame, y, groups,
                       gate_kind: str = 'gb', clf_per_modality: dict | None = None,
                       **kw):
    """
    Quantifiziert, WIE STARK das Gate seine Gewichte tatsächlich bewegt.

    Liefert je Modalität:
        mean_weight        mittleres Fusionsgewicht über alle Test-Fenster
        std_overall        Streuung über ALLE Fenster (between- + within-Patient)
        std_within_patient mittlere Streuung INNERHALB eines Patienten
                           (= echte Pro-Fenster-Anpassung; das ist der Kernwert)
        cv_within_patient  std_within_patient / mean_weight  (dimensionslos)

    Interpretation: ist std_within_patient ~ 0 (bzw. cv_within_patient << 1),
    variiert das Gewicht innerhalb eines Patienten kaum — das Gate reagiert dann
    NICHT auf die fensterweise schwankende Signalqualität, sondern wirkt faktisch
    wie eine feste (höchstens patientenweise) Gewichtung.

    Rückgabe: (report_df, weights_df)   weights_df enthält w_<mod> + patient je Fenster.
    """
    res = evaluate_moe(df, rel, y, groups, clf_per_modality=clf_per_modality,
                       gate_kind=gate_kind, return_arrays=True, return_weights=True, **kw)
    yg, yw = res[5], res[6]                      # (..., y_groups, y_weights)
    wcols = [f'w_{m}' for m in G.ORDER]
    W = pd.DataFrame(yw, columns=wcols)
    W['patient'] = yg

    mean_w   = W[wcols].mean()
    std_all  = W[wcols].std()
    std_wp   = W.groupby('patient')[wcols].std().mean()   # über Patienten gemittelt
    cv_wp    = std_wp / mean_w.replace(0, np.nan)

    rep = pd.DataFrame({
        'mean_weight':        mean_w,
        'std_overall':        std_all,
        'std_within_patient': std_wp,
        'cv_within_patient':  cv_wp,
    })
    rep.index = G.ORDER
    return rep, W


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

    # Adaptivitäts-Diagnose (gefroren vs. fensterweise) + Gewichtsstreuung
    adt = compare_adaptivity(df, rel, y, groups, base_gate='gb', inner_splits=3)
    acols = ['variant', 'AUC', 'Sensitivität', 'Spezifität', 'Accuracy', 'threshold']
    print('\n', adt[[c for c in acols if c in adt.columns]].round(3).to_string(index=False))

    rep, W = gate_weight_report(df, rel, y, groups, gate_kind='gb', inner_splits=3)
    print('\nGewichts-Report (gb):\n', rep.round(3).to_string())
    assert rep['std_within_patient'].max() > 0, 'Gate sollte auf der Synthetik fensterweise variieren'
    print('\nSelbsttest OK (gelerntes Gate sollte BCG-Rauschen abwerten und equal schlagen).')


# ──────────────────────────────────────────────────────────────────────────
# Stage 0.5 — Gate-Prädiktionsgüte: sagt der SQI das Zuverlässigkeits-Ziel überhaupt voraus?
# ──────────────────────────────────────────────────────────────────────────

def gate_predictive_validity(df: pd.DataFrame, rel: pd.DataFrame, y, groups,
                             gate_kind: str = 'gb', target_metric: str = 'cosen_err',
                             random_state: int = 42) -> pd.DataFrame:
    """
    Wurzel-Ursachen-Diagnose VOR jeder Gewichtungs-Arbeit (Stage 0.5).

    Frage: Trägt der SQI überhaupt Information über das Zuverlässigkeits-ZIEL?
    Wenn nicht, ist der prädizierte Fehler quasi konstant, die Gewichte werden
    flach, und KEINE Abbildung (Softmax oder Bachelets flat+exp) kann die Fusion
    fensterweise steuern. Misst also die UPSTREAM-Ursache; cell 15/16 messen nur
    die Downstream-Wirkung. Bachelets Fusion funktionierte, WEIL r(pred,true)=0.89.

    Vorgehen (leckagefrei, LOPO): je äußerem Fold das Gate EXAKT wie in
    `evaluate_moe` trainieren (Ziel via `_prep_targets`, 95-%-Sentinel für
    ungültige Trainingsfenster) und den ausgelassenen Patienten vorhersagen.
    Gemessen wird also der real eingesetzte Reliability-Prädiktor — kein
    geschöntes Nur-gültig-Modell.

    Bewertung NUR auf Test-Fenstern mit gültigem GT-Ziel
    (`rel_<mod>_valid == True`); Sentinel-gefüllte Fenster sind ausgeschlossen,
    sonst verzerrt die Füllung die Korrelation.

    target_metric : 'cosen_err' (AF-Default) | 'drr_sd_err' | 'hr_err'
                    (hier die ROHE Metrik-Spalte, nicht 'target').

    Rückgabe (DataFrame, index = Modalität):
        r                 Pearson-Korrelation prädizierter vs. wahrer Fehler
        rho               Spearman-Rangkorrelation (monoton, robust)
        R2_vs_train_mean  R² gegenüber der Konstanten-Baseline "Trainings-Mittel"
                          (>0 = Gate schlägt die Konstante, ≤0 = nicht besser als konstant)
        n_valid           Zahl der bewerteten Fenster (gültiges GT-Ziel)
    """
    from scipy.stats import pearsonr, spearmanr

    if target_metric not in ('cosen_err', 'drr_sd_err', 'hr_err'):
        raise ValueError("target_metric muss 'cosen_err', 'drr_sd_err' oder 'hr_err' sein "
                         "(rohe Metrik, nicht 'target').")

    y = np.asarray(y)
    groups = np.asarray(groups)
    rel_al = align_reliability(df, rel)
    gate_cols = E.gate_sqi_cols(df, 'all')
    logo = LeaveOneGroupOut()

    acc = {m: {'true': [], 'pred': [], 'base': []} for m in G.ORDER}

    for tr, te in logo.split(df, y, groups):
        # Gate exakt wie im Einsatz trainieren (sentinel-gefülltes Ziel)
        T_tr = _prep_targets(rel_al.iloc[tr], target_metric)
        gate = G.make_gate(kind=gate_kind, random_state=random_state)
        gate.fit(df.iloc[tr][gate_cols].values, T_tr)
        err_hat = np.asarray(gate.predict(df.iloc[te][gate_cols].values), dtype=float)
        tmean = T_tr.mean(axis=0)            # Konstanten-Baseline je Modalität (aus dem Training)

        rel_te = rel_al.iloc[te].reset_index(drop=True)
        for j, m in enumerate(G.ORDER):
            valid  = (rel_te[f'rel_{m}_valid'] == True).to_numpy()
            true_e = rel_te[f'rel_{m}_{target_metric}'].values.astype(float)
            ok = valid & np.isfinite(true_e)
            acc[m]['true'].append(true_e[ok])
            acc[m]['pred'].append(err_hat[ok, j])
            acc[m]['base'].append(np.full(int(ok.sum()), tmean[j]))

    rows = []
    for m in G.ORDER:
        t = np.concatenate(acc[m]['true']) if acc[m]['true'] else np.array([])
        p = np.concatenate(acc[m]['pred']) if acc[m]['pred'] else np.array([])
        b = np.concatenate(acc[m]['base']) if acc[m]['base'] else np.array([])
        n = int(len(t))
        if n >= 3 and np.std(t) > 0 and np.std(p) > 0:
            r   = float(pearsonr(p, t)[0])
            rho = float(spearmanr(p, t).correlation)
            ss_res = float(np.sum((t - p) ** 2))
            ss_tot = float(np.sum((t - b) ** 2))     # vs. Trainings-Mittel
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        else:
            r = rho = r2 = np.nan
        rows.append({'modality': m, 'r': r, 'rho': rho,
                     'R2_vs_train_mean': r2, 'n_valid': n})
    return pd.DataFrame(rows).set_index('modality')


# ──────────────────────────────────────────────────────────────────────────
# Optuna-Hyperparametersuche fürs torch_mlp-Gate (Bachelet 3.7.3 -> AF übersetzt)
# ──────────────────────────────────────────────────────────────────────────

def _smooth_l1(p, t, beta: float = 1.0) -> float:
    """SmoothL1/Huber (beta=1) als reine numpy-Funktion (kein torch nötig fürs
    Studien-Objective). Identisch zu torch.nn.SmoothL1Loss(reduction='mean')."""
    d = np.abs(np.asarray(p, float) - np.asarray(t, float))
    return float(np.mean(np.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta)))


def tune_torch_gate(df: pd.DataFrame, rel: pd.DataFrame, y, groups,
                    target_metric: str = 'target', n_trials: int = 100,
                    cv_splits: int = 5, random_state: int = 42,
                    gate_kind: str = 'torch_mlp', n_jobs: int = 1,
                    timeout: float | None = None, verbose: bool = True):
    """
    Optuna-Suche für die Gate-Hyperparameter — Bachelet 3.7.3, auf AF übersetzt.

    Zielgröße der Studie (direction='minimize') = mittlerer VALIDIERUNGS-SmoothL1
    der Fehlerprädiktion in PATIENTENGRUPPIERTER CV (GroupKFold über Patienten;
    cv_splits Folds — Bachelet nutzt volle LOPO, hier aus Laufzeitgründen K-Fold,
    aber weiterhin gruppiert => kein Patient gleichzeitig in Train/Val).
    Bewertet wird NUR auf GT-gültigen Fenstern (rel_<mod>_valid), sonst verzerrt der
    Sentinel den Verlust.

    WICHTIG (AF, nicht HF): `target_metric` ist das AF-relevante Zuverlässigkeitsziel
    (Default 'target' = die in reliability.py gewählte AF-Metrik, z.B. cosen_err/
    drr_sd_err). NICHT 'hr_err' verwenden — das wäre Bachelets HF-Ziel und nicht das
    Ziel dieser Arbeit. Das Gate wird auf die AF-relevante Zuverlässigkeit trainiert;
    die eigentliche AF-Bewertung erfolgt anschließend separat über `compare_gates`
    (fensterweise AF-Metriken, LOPO).

    Suchraum (wie Bachelet Tab. 3.11): hidden_dims {64,128,(128,64),(256,128)},
    lr [1e-4,8e-3] log, weight_decay [2e-5,2e-3] log, batch_size {32,64,128,256},
    max_epochs [20,600], patience [8,60], dropout [0,0.3].

    Hinweis Parallelität: bei n_jobs>1 ggf. torch.set_num_threads(1) setzen, damit
    Optuna-Trials und torch-Intraop-Threads sich nicht überzeichnen.

    Rückgabe: (best_params: dict, study). best_params lässt sich direkt als
    `gate_params=...` an evaluate_moe / compare_gates übergeben.
    """
    import optuna
    from sklearn.model_selection import GroupKFold

    if target_metric == 'hr_err':
        raise ValueError("target_metric='hr_err' ist Bachelets HF-Ziel — diese Arbeit "
                         "detektiert AF. Nutze 'target' (= AF-Metrik) bzw. 'cosen_err'/"
                         "'drr_sd_err'.")

    y = np.asarray(y); groups = np.asarray(groups)
    rel_al = align_reliability(df, rel)
    gate_cols = E.gate_sqi_cols(df, 'all')
    Xall = df[gate_cols].values

    n_splits = min(cv_splits, len(np.unique(groups)))
    folds = list(GroupKFold(n_splits=n_splits).split(df, y, groups))

    def objective(trial):
        hd = trial.suggest_categorical('hidden_dims', ['64', '128', '128,64', '256,128'])
        hp = dict(
            hidden_dims=tuple(int(x) for x in hd.split(',')),
            lr=trial.suggest_float('lr', 1e-4, 8e-3, log=True),
            weight_decay=trial.suggest_float('weight_decay', 2e-5, 2e-3, log=True),
            batch_size=trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            max_epochs=trial.suggest_int('max_epochs', 20, 600),
            patience=trial.suggest_int('patience', 8, 60),
            dropout=trial.suggest_float('dropout', 0.0, 0.3),
        )
        losses = []
        for tr, va in folds:
            T_tr = _prep_targets(rel_al.iloc[tr], target_metric)
            gate = G.make_gate(kind=gate_kind, random_state=random_state, **hp)
            gate.fit(Xall[tr], T_tr)
            pred = np.asarray(gate.predict(Xall[va]), dtype=float)
            rel_va = rel_al.iloc[va].reset_index(drop=True)
            p_all, t_all = [], []
            for j, m in enumerate(G.ORDER):
                valid = (rel_va[f'rel_{m}_valid'] == True).to_numpy()
                true_e = rel_va[f'rel_{m}_{target_metric}'].values.astype(float)
                ok = valid & np.isfinite(true_e)
                if ok.sum() == 0:
                    continue
                p_all.append(pred[ok, j]); t_all.append(true_e[ok])
            if p_all:
                losses.append(_smooth_l1(np.concatenate(p_all), np.concatenate(t_all)))
        return float(np.mean(losses)) if losses else float('inf')

    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout)

    best = dict(study.best_params)
    best['hidden_dims'] = tuple(int(x) for x in best['hidden_dims'].split(','))
    if verbose:
        print(f'Beste Val-SmoothL1: {study.best_value:.4f}  ·  beste Params: {best}')
    return best, study
