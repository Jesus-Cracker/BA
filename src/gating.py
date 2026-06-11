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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin

ORDER = ['cecg', 'ppg', 'bcg']   # feste Modalitäts-Reihenfolge


# ──────────────────────────────────────────────────────────────────────────
# PyTorch-MLP-Gate (Bachelet-Architektur 3.6.1), als sklearn-kompatibler
# Multi-Output-Regressor. torch wird LAZY importiert, damit gating.py auch ohne
# torch importierbar bleibt (gb/xgb/mlp/ridge funktionieren weiter).
# ──────────────────────────────────────────────────────────────────────────

class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """
    Vollverbundenes MLP (PyTorch): SQI-Vektor -> [err_cecg, err_ppg, err_bcg].

    Bachelet-Stil (Fehlerprädiktion 3.6.1): ReLU in den versteckten Schichten,
    LINEARE Ausgabe (Regression), SmoothL1Loss (robust ggü. Ausreißer-Fehlern),
    Adam, L2 via `weight_decay`, Early-Stopping auf internem Validierungssplit.

    WICHTIG (AF, nicht HF): Lernziel ist der AF-relevante Zuverlässigkeitsfehler
    (z.B. dRR_SD-/CoSEn-Fehler aus reliability.py). Nur die ARCHITEKTUR stammt von
    Bachelet (HF-Schätzung) — das Ziel bleibt AF.

    Ein-/Ausgaben werden hier NICHT skaliert: das übernimmt die umgebende Pipeline
    (StandardScaler) bzw. der TransformedTargetRegressor in `make_gate` — exakt wie
    bei den übrigen Gates, also kein doppeltes Normalisieren und leckagefrei pro Fold.
    """

    def __init__(self, hidden_dims=(128, 64), lr: float = 1e-3,
                 weight_decay: float = 1e-4, batch_size: int = 128,
                 max_epochs: int = 300, patience: int = 20, dropout: float = 0.0,
                 val_frac: float = 0.15, random_state: int = 42, device: str = 'cpu'):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.dropout = dropout
        self.val_frac = val_frac
        self.random_state = random_state
        self.device = device

    def _build_net(self, n_in: int, n_out: int):
        import torch.nn as nn
        dims = list(self.hidden_dims) if self.hidden_dims else []
        layers, d = [], n_in
        for h in dims:
            layers += [nn.Linear(d, int(h)), nn.ReLU()]
            if self.dropout and float(self.dropout) > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            d = int(h)
        layers.append(nn.Linear(d, n_out))      # lineare Ausgabe (Regression)
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch
        torch.manual_seed(int(self.random_state))
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        self.n_features_in_ = int(X.shape[1])
        self.n_outputs_ = int(y.shape[1])
        dev = torch.device(self.device)

        # interner Train/Val-Split fürs Early-Stopping (nur innerhalb der Trainingsdaten)
        rng = np.random.default_rng(int(self.random_state))
        n = len(X)
        idx = rng.permutation(n)
        n_val = int(round(float(self.val_frac) * n)) if self.val_frac else 0
        n_val = min(max(n_val, 0), max(n - 1, 0))
        va_idx, tr_idx = idx[:n_val], idx[n_val:]
        use_es = n_val >= 1 and len(tr_idx) >= 1

        Xtr = torch.from_numpy(X[tr_idx]).to(dev)
        ytr = torch.from_numpy(y[tr_idx]).to(dev)
        if use_es:
            Xva = torch.from_numpy(X[va_idx]).to(dev)
            yva = torch.from_numpy(y[va_idx]).to(dev)

        self.net_ = self._build_net(self.n_features_in_, self.n_outputs_).to(dev)
        opt = torch.optim.Adam(self.net_.parameters(), lr=float(self.lr),
                               weight_decay=float(self.weight_decay))
        loss_fn = torch.nn.SmoothL1Loss()

        bs = max(int(self.batch_size), 1)
        best_val, best_state, bad = np.inf, None, 0
        gen = torch.Generator().manual_seed(int(self.random_state))
        for _ in range(int(self.max_epochs)):
            self.net_.train()
            perm = torch.randperm(len(Xtr), generator=gen)
            for s in range(0, len(Xtr), bs):
                b = perm[s:s + bs]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
            if use_es:
                self.net_.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(self.net_(Xva), yva).item())
                if vloss < best_val - 1e-6:
                    best_val, bad = vloss, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in self.net_.state_dict().items()}
                else:
                    bad += 1
                    if bad >= int(self.patience):
                        break
        if use_es and best_state is not None:
            self.net_.load_state_dict(best_state)
        self.best_val_loss_ = float(best_val) if use_es else float('nan')
        return self

    def predict(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.from_numpy(X).to(torch.device(self.device))).cpu().numpy()
        return out.ravel() if self.n_outputs_ == 1 else out


def _target_transformer(target_transform: str = 'standard'):
    """Ziel-Transformer für den TransformedTargetRegressor des Gates.

    Motiviert durch rho > r in der Prädiktionsgüte (monotoner, aber nicht-linearer
    Zusammenhang SQI->Fehler) und durch die schwere Schiefe der Fehlerverteilungen:
      'standard'  StandardScaler (bisheriges Verhalten, unverändert).
      'quantile'  QuantileTransformer (rangbasiert -> Normalverteilung). Lernt auf
                  den RÄNGEN des Fehlers -> nutzt genau die monotone Struktur, die
                  rho zeigt; robust gegen Ausreißer/Heavy-Tails.
      'log'       log1p dann StandardScaler — komprimiert Heavy-Tails (Fehler >= 0),
                  inverse via expm1.
    Die Rücktransformation (predict) liefert in allen Fällen Werte in Original-Fehler-
    einheiten zurück -> die nachgelagerte Fehler->Gewicht-Abbildung bleibt unverändert.
    """
    if target_transform == 'standard':
        return StandardScaler()
    if target_transform == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        return QuantileTransformer(output_distribution='normal',
                                   n_quantiles=256, subsample=100_000,
                                   random_state=0)
    if target_transform == 'log':
        from sklearn.preprocessing import FunctionTransformer
        return FunctionTransformer(func=np.log1p, inverse_func=np.expm1,
                                   check_inverse=False)
    raise ValueError("target_transform muss 'standard', 'quantile' oder 'log' sein")


def make_gate(kind: str = 'mlp', hidden=(64, 32), alpha: float = 1e-3,
              random_state: int = 42, target_transform: str = 'standard', **gate_kw):
    """
    Multi-Output-Regressor: SQI-Vektor -> [err_cecg, err_ppg, err_bcg].
    Ein- und Ausgaben werden normalisiert (Ziel via TransformedTargetRegressor).

    target_transform : Ziel-Transformation ('standard'|'quantile'|'log'), s.
        _target_transformer. 'quantile' nutzt die monotone (rho>r) Struktur direkt.
    """
    ttf = _target_transformer(target_transform)
    if kind == 'mlp':
        reg = MLPRegressor(hidden_layer_sizes=hidden, activation='relu',
                           alpha=alpha, max_iter=800, random_state=random_state)
    elif kind == 'gb':
        reg = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=random_state))
    elif kind == 'ridge':
        reg = Ridge(alpha=1.0, random_state=random_state)
    elif kind == 'xgb':
        # XGBoost-Gate: gleiche Familie wie 'gb' (Gradient-Boosting-Bäume).
        # Hier nur, um Nicos Vorschlag empirisch zu prüfen — erwartungsgemäß
        # nahe an 'gb', da das Gate informations- und nicht modellbegrenzt ist.
        reg = MultiOutputRegressor(XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state,
            n_jobs=1, tree_method='hist'))
    elif kind == 'torch_mlp':
        # PyTorch-MLP-Gate (Bachelet-Architektur 3.6.1). Hyperparameter kommen aus
        # gate_kw (Optuna-Resultat); Defaults = Bachelets finale Wahl (128,64).
        # Lernziel bleibt der AF-relevante Zuverlässigkeitsfehler, NICHT HF.
        return TransformedTargetRegressor(
            regressor=Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('sc',  StandardScaler()),
                ('reg', TorchMLPRegressor(
                    hidden_dims=gate_kw.get('hidden_dims', (128, 64)),
                    lr=gate_kw.get('lr', 1e-3),
                    weight_decay=gate_kw.get('weight_decay', 1e-4),
                    batch_size=gate_kw.get('batch_size', 128),
                    max_epochs=gate_kw.get('max_epochs', 300),
                    patience=gate_kw.get('patience', 20),
                    dropout=gate_kw.get('dropout', 0.0),
                    val_frac=gate_kw.get('val_frac', 0.15),
                    random_state=random_state))]),
            transformer=ttf)
    else:
        raise ValueError("kind muss 'mlp', 'gb', 'ridge', 'xgb' oder 'torch_mlp' sein")

    inner = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc',  StandardScaler()),
                      ('reg', reg)])
    # Ziel ebenfalls normalisieren (Fehlerwerte verschiedener Modalitäten skalieren unterschiedlich)
    return TransformedTargetRegressor(regressor=inner, transformer=ttf)


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


def blend_weights(w_gate, gate_trust, eps: float = 1e-6):
    """Blendet eine fertige Gewichtsmatrix modalitätsweise Richtung Gleichgewicht.

    Motivation (erklärt den PAT019-Fehlermode): sagt das Gate den Zuverlässigkeits-
    fehler einer Modalität NICHT vorher (R² ~ 0), ist ihr err_hat praktisch Rauschen
    und das daraus berechnete Gewicht zufällig — die Fusion merkt das nicht. Statt
    einem Rausch-Gewicht zu vertrauen, wird je Modalität mit dem Gleichgewicht (1/K)
    gemischt, gewichtet mit dem VERTRAUEN (geclipptes Gate-R² je Modalität):

        w = trust ⊙ w_gate + (1 − trust) ⊙ w_equal   (danach zeilenweise renormiert)

    trust=1 -> volles Gate-Gewicht; trust=0 -> Gleichgewicht (graceful degradation).
    gate_trust: Array der Länge K (Modalitäten in ORDER), aus leckagefreiem R².
    """
    w_gate = np.asarray(w_gate, dtype=float)
    if w_gate.ndim == 1:
        w_gate = w_gate[None, :]
    K = w_gate.shape[1]
    w_equal = np.full_like(w_gate, 1.0 / K)
    trust = np.clip(np.asarray(gate_trust, dtype=float), 0.0, 1.0).reshape(1, -1)
    w = trust * w_gate + (1.0 - trust) * w_equal
    return w / (w.sum(axis=1, keepdims=True) + eps)


def errors_to_weights_blended(err_hat, gate_trust, scale: float | None = None,
                              eps: float = 1e-6):
    """Softmax-Fehler->Gewicht (errors_to_weights) + R²-Blend Richtung Gleichgewicht.
    Bequemer Einzelaufruf; identisch zu blend_weights(errors_to_weights(...), trust)."""
    return blend_weights(errors_to_weights(err_hat, scale=scale, eps=eps),
                         gate_trust, eps=eps)


def errors_to_weights_exp(err_hat, e0: float, tau: float, eps: float = 1e-6):
    """
    Bachelet-Abbildung (3.6.2) Fehler -> Gewicht, zeilenweise normiert (Σ=1):

        SQI = 1                        für err <= e0   (Toleranz für kleine Fehler)
        SQI = exp(-(err - e0) / tau)   für err >  e0   (Abklingen mit Konstante τ)

    Anschließend werden die SQIs je Fenster auf Σ=1 normiert -> Fusionsgewichte.
    Ungültige (NaN) Fehler -> SQI 0 (maximal unzuverlässig). Ist die ganze Zeile
    ungültig, wird auf Gleichgewicht zurückgefallen (statt Division durch 0).

    e0, τ werden wie bei Bachelet per Optuna bestimmt (hier explizit übergeben).
    Kleines τ = härtere, fast Argmin-artige Gewichtung; großes τ = weicher.
    Monoton fallend wie `errors_to_weights`, nur mit Toleranzschwelle e0 und
    exponentiellem statt softmax-Abfall.
    """
    err = np.asarray(err_hat, dtype=float)
    if err.ndim == 1:
        err = err[None, :]
    err = np.where(np.isfinite(err), np.maximum(err, 0.0), err)  # Fehler semantisch >= 0
    tau = max(float(tau), eps)
    sqi = np.where(err <= e0, 1.0, np.exp(-(err - e0) / tau))
    sqi = np.where(np.isfinite(err), sqi, 0.0)
    s = sqi.sum(axis=1, keepdims=True)
    return np.where(s > eps, sqi / (s + eps), 1.0 / err.shape[1])


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

    # Bachelet-Abbildung (e0/τ): kleiner Fehler -> Gewicht ~1, großer -> ~0, Σ=1
    we = errors_to_weights_exp(err_hat, e0=0.05, tau=0.5)
    print('exp-Gewichtssummen ~1 :', np.allclose(we.sum(axis=1), 1.0))
    print('exp-mittlere Gewichte :', {m: round(we[:, i].mean(), 3) for i, m in enumerate(ORDER)})
    assert we[:, 1].mean() > we[:, 2].mean(), 'exp: PPG (kleiner Fehler) muss > BCG gewichtet sein'

    # torch_mlp-Gate nur testen, wenn PyTorch installiert ist (sonst überspringen)
    try:
        import torch  # noqa: F401
        gate_t = make_gate(kind='torch_mlp', max_epochs=60, patience=15, random_state=0)
        gate_t.fit(SQI[:split], err_true[:split])
        err_hat_t = gate_t.predict(SQI[split:])
        print('torch_mlp R² je Modalität:',
              {m: round(r2_score(err_true[split:, i], err_hat_t[:, i]), 3) for i, m in enumerate(ORDER)})
        wt = errors_to_weights(err_hat_t)
        assert np.allclose(wt.sum(axis=1), 1.0)
        print('torch_mlp-Gate OK.')
    except ImportError:
        print('torch nicht installiert -> torch_mlp-Test übersprungen (erwartet in diesem Sandbox).')

    print('Selbsttest OK.')
