#!/usr/bin/env python3
# Fani, Akinloye, Liu & Xu, University of Oklahoma (2026)
# UHTS surrogate model for Fe80Cr16Ni4 nanocrystals

import os
import numpy as np
import pandas as pd
import warnings
import optuna
import shap
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rcParams
from torch.utils.data import DataLoader, TensorDataset
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RepeatedKFold, KFold, GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
rcParams["font.family"]   = "DejaVu Sans"
rcParams["font.size"]     = 13
rcParams["axes.linewidth"] = 1.0
rcParams["axes.labelpad"] = 8

SEED           = 42
QUICK          = False
N_JOBS         = 1
N_MODELS_FINAL = 10
N_MODELS_CV    = 5
N_TRIALS_INNER = 10
INNER_SPLITS   = 3
N_SPLITS_OUTER = 10
N_REPEATS      = 5
GKF_SPLITS     = 5
TTA_PASSES     = 1
SHAP_NSAMPLES  = 150
SHAP_BG        = 10
CV_ABLATION    = True
N_BOOTSTRAP    = 2000
ARCH_CHOICES   = [(32, 16), (48, 24), (64, 32), (64, 64)]
GRAIN_C, VOID_C = "#1f77b4", "#d62728"

if QUICK:
    N_MODELS_FINAL, N_MODELS_CV, N_TRIALS_INNER = 2, 2, 3
    INNER_SPLITS, N_REPEATS, GKF_SPLITS = 2, 1, 2
    TTA_PASSES, SHAP_NSAMPLES, SHAP_BG = 1, 40, 5
    N_BOOTSTRAP = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


# =====================================================================
#  1. Dataset
# =====================================================================
VOID_FREE = [
    (10,  "Equiaxed", [4883.738, 4910.627, 5000.630]),
    (12.5,"Equiaxed", [4716.600, 5230.700, 5136.600]),
    (15,  "Equiaxed", [5658.105, 5563.383, 5686.377]),
    (17.5,"Equiaxed", [5959.000, 5646.000, 5970.000]),
    (20,  "Equiaxed", [4743.231, 4990.020, 5010.780]),
    (10,  "Columnar", [5195.072, 5349.290, 5612.630]),
    (12.5,"Columnar", [5545.900, 5310.100, 4638.300]),
    (15,  "Columnar", [5559.355, 5486.398, 5424.627]),
    (17.5,"Columnar", [5401.200, 5812.000, 5559.100]),
    (20,  "Columnar", [4516.982, 4790.460, 4980.750]),
]
SPH_INT = [
    (10,  "Equiaxed", [3503.188,3354.142,3419.141,3368.555,3390.060]),
    (12.5,"Equiaxed", [3443.900,3258.400,3249.700,3499.000,3486.200]),
    (15,  "Equiaxed", [3959.253,4022.380,4110.265,3939.886,3960.746]),
    (17.5,"Equiaxed", [4162.000,4046.000,4046.000,4037.000,3881.000]),
    (20,  "Equiaxed", [3585.079,3640.643,3566.088,3568.024,3495.653]),
    (10,  "Columnar", [4197.749,4129.078,4567.889,4331.483,4298.854]),
    (12.5,"Columnar", [4507.300,4162.300,3738.500,4390.200,4423.900]),
    (15,  "Columnar", [4374.370,4119.882,4028.399,4357.496,4357.496]),
    (17.5,"Columnar", [4604.200,4578.200,4600.800,4534.000,4470.800]),
    (20,  "Columnar", [3822.848,3859.674,3834.406,3603.685,3473.315]),
]
SPH_GB = [
    (10,  "Equiaxed", [3245.698,3367.360,3390.535,3474.526,3342.819]),
    (12.5,"Equiaxed", [3416.800,3285.400,3272.700,3372.000,3500.900]),
    (15,  "Equiaxed", [4110.265,3993.455,4112.391,3922.710,3991.429]),
    (17.5,"Equiaxed", [3897.000,4133.000,4040.000,3957.000,4157.000]),
    (20,  "Equiaxed", [3585.079,3734.433,3679.311,3516.168,3476.085]),
    (10,  "Columnar", [4147.648,4288.204,3837.379,4140.089,4233.086]),
    (12.5,"Columnar", [3982.700,3899.900,3405.900,3688.100,3921.600]),
    (15,  "Columnar", [4219.584,3999.552,4010.205,3859.290,3860.842]),
    (17.5,"Columnar", [4372.800,4415.100,4482.100,4387.100,4301.800]),
    (20,  "Columnar", [3473.315,3800.727,3274.761,3406.695,3402.717]),
]
IRR_GB = [
    (10,  "Equiaxed", [3269.777,3147.577,3340.144,3340.144,3273.410]),
    (12.5,"Equiaxed", [3329.100,3224.800,3251.400,3363.700,3276.200]),
    (15,  "Equiaxed", [3822.103,3959.374,3860.706,3956.963,3836.096]),
    (17.5,"Equiaxed", [3959.000,3859.000,3651.000,3602.000,4040.000]),
    (20,  "Equiaxed", [3581.370,3406.695,3402.717,3516.168,3353.806]),
    (10,  "Columnar", [4164.610,4150.085,4120.846,3998.664,4328.446]),
    (12.5,"Columnar", [3846.300,3868.000,4092.600,3910.300,3827.600]),
    (15,  "Columnar", [3830.150,3859.680,3727.930,3842.324,3965.279]),
    (17.5,"Columnar", [4320.100,4190.300,4162.800,4231.500,4252.800]),
    (20,  "Columnar", [3676.162,3646.863,3015.406,3392.151,3338.219]),
]

_rows = []
for gs, sh, vals in VOID_FREE:
    for i, v in enumerate(vals, start=1):
        _rows.append((gs, sh, "None", "None", v, i))
for src, vshape, vloc in [(SPH_INT, "Spherical", "Interior"),
                          (SPH_GB,  "Spherical", "Grain Boundary"),
                          (IRR_GB,  "Irregular", "Grain Boundary")]:
    for gs, sh, vals in src:
        for v in vals:
            _rows.append((gs, sh, vshape, vloc, v, 1))

df = pd.DataFrame(_rows, columns=["Grain Size (nm)", "Grain Shape", "Void Shape",
                                  "Void Location", "UHTS (MPa)", "vf_rep"])
df["Void Presence"] = (df["Void Shape"] != "None").astype(float)
df["family"]    = df["Grain Shape"] + "_" + df["Grain Size (nm)"].astype(str)
df["parent_id"] = np.where(df["vf_rep"] == 1, df["family"],
                           df["family"] + "_vf" + df["vf_rep"].astype(str))

y = df["UHTS (MPa)"].values.astype(np.float32)
family_groups = df["family"].values
parent_groups = df["parent_id"].values


# =====================================================================
#  2. Feature construction
# =====================================================================
def build_Xy(frame, level):
    gs = frame["Grain Size (nm)"].values.astype(np.float32)
    eq = (frame["Grain Shape"] == "Equiaxed").astype(np.float32).values
    if level == "pred":
        vsi = (frame["Void Shape"] == "Irregular").astype(np.float32).values
        vss = (frame["Void Shape"] == "Spherical").astype(np.float32).values
        vlg = (frame["Void Location"] == "Grain Boundary").astype(np.float32).values
        vli = (frame["Void Location"] == "Interior").astype(np.float32).values
        X = np.column_stack([gs, eq, vsi, vss, vlg, vli])
        names = ["Grain Size", "Grain Shape_Equiaxed", "Void Shape_Irregular",
                 "Void Shape_Spherical", "Void Location_GB", "Void Location_Interior"]
    elif level == "L1":
        X = np.column_stack([gs, eq, frame["Void Presence"].values.astype(np.float32)])
        names = ["Grain Size", "Grain Shape_Equiaxed", "Void Presence"]
    elif level == "shape":
        vsi = (frame["Void Shape"] == "Irregular").astype(np.float32).values
        X = np.column_stack([gs, eq, vsi])
        names = ["Grain Size", "Grain Shape_Equiaxed", "Void Shape_Irregular"]
    elif level == "loc":
        vlg = (frame["Void Location"] == "Grain Boundary").astype(np.float32).values
        X = np.column_stack([gs, eq, vlg])
        names = ["Grain Size", "Grain Shape_Equiaxed", "Void Location_GB"]
    return X.astype(np.float32), names

def scale_col0(train_col0, *mats):
    m, s = float(train_col0.mean()), float(train_col0.std() or 1.0)
    return [np.column_stack([(M[:, 0] - m) / s, M[:, 1:]]).astype(np.float32) for M in mats]


# =====================================================================
#  3. Residual DNN
# =====================================================================
class ResidualBlock(nn.Module):
    def __init__(self, i, o, dropout, hnoise):
        super().__init__()
        self.hnoise = hnoise
        self.lin = nn.Linear(i, o)
        self.bn  = nn.BatchNorm1d(o)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(i, o, bias=False) if i != o else nn.Identity()
    def forward(self, x):
        out = self.bn(self.lin(x))
        if self.training and self.hnoise > 0:
            out = out + torch.randn_like(out) * self.hnoise
        return self.act(self.drop(out)) + self.skip(x)

def build_net(n_in, hidden, dropout, hnoise):
    blocks, i = [], n_in
    for h in hidden:
        blocks.append(ResidualBlock(i, h, dropout, hnoise)); i = h
    blocks.append(nn.Linear(i, 1))
    return nn.Sequential(*blocks)

class DNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=(48, 24), dropout=0.2, lr=1e-3, wd=1e-4, bs=16,
                 epochs=600, patience=40, in_noise=0.01, h_noise=0.01,
                 tta=TTA_PASSES, log_target=True):
        self.hidden=hidden; self.dropout=dropout; self.lr=lr; self.wd=wd; self.bs=bs
        self.epochs=epochs; self.patience=patience; self.in_noise=in_noise
        self.h_noise=h_noise; self.tta=tta; self.log_target=log_target
    def _fwd(self, y):
        yt = np.log(y) if self.log_target else y
        self.ym = float(yt.mean()); self.ys = float(yt.std() or 1.0)
        return (yt - self.ym) / self.ys
    def _inv(self, z):
        yt = z * self.ys + self.ym
        return np.exp(yt) if self.log_target else yt
    def _noise(self, x):
        if self.in_noise <= 0:
            return x
        n = torch.zeros_like(x)
        n[:, [0]] = torch.randn(x.shape[0], 1, device=x.device) * self.in_noise
        return x + n
    def fit(self, X, y, groups=None):
        Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        yt = torch.tensor(self._fwd(y), dtype=torch.float32).to(DEVICE)
        n = len(Xt)
        vi = ti = None
        if groups is not None and len(np.unique(groups)) >= 4:
            uniq = np.unique(groups)
            k = max(1, int(round(len(uniq) * 0.2)))
            val_g = np.random.choice(uniq, k, replace=False)
            mask = np.isin(groups, val_g)
            vi, ti = np.where(mask)[0], np.where(~mask)[0]
            if len(vi) == 0 or len(ti) < 4:
                vi = ti = None
        if vi is None:
            nv = max(1, int(n * 0.15))
            idx = np.random.permutation(n); vi, ti = idx[:nv], idx[nv:]
        self.net_ = build_net(Xt.shape[1], self.hidden, self.dropout, self.h_noise).to(DEVICE)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.MSELoss()
        bs = min(self.bs, max(2, len(ti) // 4))
        loader = DataLoader(TensorDataset(Xt[ti], yt[ti]), batch_size=bs,
                            shuffle=True, drop_last=True)
        best, best_w, wait = float("inf"), None, 0
        for _ in range(self.epochs):
            self.net_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(self.net_(self._noise(xb)).squeeze(-1), yb).backward()
                opt.step()
            sch.step(); self.net_.eval()
            with torch.no_grad():
                vl = loss_fn(self.net_(Xt[vi]).squeeze(-1), yt[vi]).item()
            if vl < best - 1e-5:
                best, wait = vl, 0
                best_w = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_w:
            self.net_.load_state_dict(best_w)
        self.net_.eval(); return self
    def predict(self, X, use_tta=False):
        self.net_.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            if use_tta and self.tta > 1:
                z = np.stack([self.net_(self._noise(Xt)).squeeze(-1).cpu().numpy()
                              for _ in range(self.tta)]).mean(0)
            else:
                z = self.net_(Xt).squeeze(-1).cpu().numpy()
        return self._inv(z)

class EnsembleDNN(BaseEstimator, RegressorMixin):
    def __init__(self, n_models=15, **kw): self.n_models=n_models; self.kw=kw
    def fit(self, X, y, groups=None):
        self.models_ = []
        for s in range(self.n_models):
            torch.manual_seed(s); np.random.seed(s)
            self.models_.append(DNNRegressor(**self.kw).fit(X, y, groups=groups))
        return self
    def predict(self, X, use_tta=False):
        return np.stack([m.predict(X, use_tta=use_tta) for m in self.models_]).mean(0)


# =====================================================================
#  4. Nested tuning + CV runners + baselines
# =====================================================================
def tune(Xtr, ytr, inner_groups=None):
    if inner_groups is not None and len(np.unique(inner_groups)) >= 2:
        splitter = GroupKFold(n_splits=min(INNER_SPLITS, len(np.unique(inner_groups))))
        gen_split = lambda: splitter.split(Xtr, ytr, inner_groups)
    else:
        splitter = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
        gen_split = lambda: splitter.split(Xtr)
    def objective(t):
        kw = dict(hidden=ARCH_CHOICES[t.suggest_int("arch", 0, len(ARCH_CHOICES) - 1)],
                  dropout=t.suggest_float("dropout", 0.05, 0.40),
                  lr=t.suggest_float("lr", 5e-4, 6e-3, log=True),
                  wd=t.suggest_float("wd", 1e-6, 1e-2, log=True),
                  in_noise=t.suggest_float("in_noise", 0.0, 0.05),
                  h_noise=t.suggest_float("h_noise", 0.0, 0.05),
                  patience=t.suggest_int("patience", 25, 70),
                  bs=t.suggest_categorical("bs", [8, 16, 32]),
                  epochs=600, tta=5, log_target=True)
        scores = []
        for tr, va in gen_split():
            Xs, = scale_col0(Xtr[tr][:, 0], Xtr[tr])
            Xv, = scale_col0(Xtr[tr][:, 0], Xtr[va])
            g_tr = inner_groups[tr] if inner_groups is not None else None
            m = DNNRegressor(**kw).fit(Xs, ytr[tr], groups=g_tr)
            scores.append(r2_score(ytr[va], m.predict(Xv)))
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS_INNER)
    bp = study.best_params
    return dict(hidden=ARCH_CHOICES[bp.pop("arch")], epochs=600,
                tta=TTA_PASSES, log_target=True, **bp)

def metrics(yt, yp):
    return dict(R2=r2_score(yt, yp), RMSE=np.sqrt(mean_squared_error(yt, yp)),
                MAE=mean_absolute_error(yt, yp),
                MAPE=np.mean(np.abs((yt - yp) / yt)) * 100)

def _fold_worker(tr, te, inner_groups):
    if DEVICE.type == "cpu":
        torch.set_num_threads(1)
    X, _ = build_Xy(df, "pred")
    hp = tune(X[tr], y[tr], inner_groups=inner_groups)
    Xs, Xv = scale_col0(X[tr][:, 0], X[tr], X[te])
    ens = EnsembleDNN(n_models=N_MODELS_CV, **hp).fit(Xs, y[tr], groups=inner_groups)
    pr = ens.predict(Xv); m = metrics(y[te], pr)
    return list(te), list(y[te]), list(pr), m

def run_cv(splitter, groups=None, group_tuning=False):
    X, _ = build_Xy(df, "pred")
    folds = list(splitter.split(X, y, groups) if groups is not None else splitter.split(X))
    res = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_fold_worker)(tr, te, groups[tr] if (group_tuning and groups is not None) else None)
        for tr, te in folds)
    te_idx, yt, yp, per_fold = [], [], [], {"R2": [], "RMSE": [], "MAE": [], "MAPE": []}
    for idx, a, b, m in res:
        te_idx += idx; yt += a; yp += b
        for k in per_fold:
            per_fold[k].append(m[k])
    out = metrics(np.array(yt), np.array(yp))
    for k in per_fold:
        out[f"fold_{k}_mean"] = float(np.mean(per_fold[k]))
        out[f"fold_{k}_sd"]   = float(np.std(per_fold[k]))
    out["n_pred"] = len(yt)
    out["oof"] = (np.array(te_idx), np.array(yt, float), np.array(yp, float))
    return out

def parent_bootstrap_ci(oof, sample_groups, n_boot=N_BOOTSTRAP, seed=SEED):
    te_idx, yt, yp = oof
    g = sample_groups[te_idx]
    uniq = np.unique(g)
    by_parent = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.RandomState(seed)
    acc = {k: [] for k in ("R2", "RMSE", "MAE", "MAPE")}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([by_parent[p] for p in pick])
        m = metrics(yt[rows], yp[rows])
        for k in acc:
            acc[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in acc.items()}

def baseline_cv(model, splitter, groups=None):
    X, _ = build_Xy(df, "pred"); yt, yp = [], []
    it = splitter.split(X, y, groups) if groups is not None else splitter.split(X)
    for tr, te in it:
        Xs, Xv = scale_col0(X[tr][:, 0], X[tr], X[te])
        model.fit(Xs, y[tr]); yt += list(y[te]); yp += list(model.predict(Xv))
    return metrics(np.array(yt), np.array(yp))


# =====================================================================
#  5. SHAP + figures
# =====================================================================
def _named(imp):
    out = {}
    for k, v in imp.items():
        base = ("Grain Size" if "Grain Size" in k else
                "Grain Morphology" if ("Shape" in k and "Void" not in k) else
                "Void Presence" if "Presence" in k else
                "Void Shape" if "Shape" in k else
                "Void Location" if "Location" in k else k)
        out[base] = out.get(base, 0.0) + v
    return out

def shap_analysis(predict_fn, Xs, names, nsamples=SHAP_NSAMPLES, bg_seed=0):
    rng = np.random.RandomState(bg_seed)
    bg = Xs[rng.choice(len(Xs), min(SHAP_BG, len(Xs)), replace=False)]
    sv = shap.KernelExplainer(predict_fn, bg).shap_values(Xs, nsamples=nsamples, silent=True)
    return _named(dict(zip(names, np.abs(sv).mean(0)))), sv

def fit_ensemble(frame, level, hp):
    X, n = build_Xy(frame, level)
    Xs, = scale_col0(X[:, 0], X)
    ens = EnsembleDNN(n_models=N_MODELS_FINAL, **hp).fit(
        Xs, frame["UHTS (MPa)"].values.astype(np.float32))
    return ens, Xs, n

def _hbar(ax, g, title):
    order = sorted(g, key=lambda k: g[k]); vals = [g[k] for k in order]
    clr = [VOID_C if "Void" in k else GRAIN_C for k in order]; mx = max(vals)
    bars = ax.barh(range(len(order)), vals, height=0.55, color=clr, alpha=0.92,
                   edgecolor="black", linewidth=0.5, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + mx * 0.025, b.get_y() + b.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=17, fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=17)
    ax.set_xlabel("Mean |SHAP value| (MPa)", fontsize=19, labelpad=10)
    ax.set_xlim(0, mx * 1.32)
    ax.tick_params(axis="x", labelsize=16, length=4)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, ls="--", lw=0.6, color="0.80", zorder=1)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)

def make_figures(g1, gshape, gloc):
    # Fig 9a - Level-1 SHAP bar (void presence vs grain descriptors)
    fig, ax = plt.subplots(figsize=(9.0, 4.5), dpi=300)
    _hbar(ax, g1, "Level 1: void presence vs grain descriptors")
    ax.legend(
        handles=[Patch(facecolor=GRAIN_C, label="Grain feature"),
                 Patch(facecolor=VOID_C,  label="Void feature")],
        fontsize=15,
        title="Feature group",
        title_fontsize=15,
        loc="lower right",
        framealpha=0.85,
        edgecolor="0.75",
    )
    plt.tight_layout(pad=1.5)
    plt.savefig("shap_level1_bar.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    # Fig 9b - Controlled-shape / controlled-location bar (side-by-side)
    fig, axs = plt.subplots(1, 2, figsize=(15.0, 5.0), dpi=300)
    _hbar(axs[0], gshape, "Controlled shape (GB voids; void location fixed)")
    _hbar(axs[1], gloc,   "Controlled location (spherical voids; void shape fixed)")
    axs[0].set_title("GB voids\n(void location fixed)", fontsize=16, pad=8, loc="left")
    axs[1].set_title("Spherical voids\n(void shape fixed)",  fontsize=16, pad=8, loc="left")
    axs[1].legend(
        handles=[Patch(facecolor=GRAIN_C, label="Grain feature"),
                 Patch(facecolor=VOID_C,  label="Void feature")],
        fontsize=15,
        title="Feature group",
        title_fontsize=15,
        loc="lower right",
        framealpha=0.85,
        edgecolor="0.75",
    )
    plt.tight_layout(pad=1.5, w_pad=3.5)
    plt.savefig("shap_controlled_bar.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def _order(g):
    return " > ".join(sorted(g, key=lambda k: -g[k]))

def _report(title, g):
    tot = sum(g.values())
    print(f"\n{title}\n  {_order(g)}")
    for k in sorted(g, key=lambda k: -g[k]):
        print(f"    {k:<18}{g[k]:>8.1f} MPa   share {g[k] / tot * 100:5.1f}%")


# =====================================================================
#  6. Main
# =====================================================================
def main():
    n_vf = int((df["Void Shape"] == "None").sum())
    print(f"Dataset: {len(y)} samples ({n_vf} void-free + {len(y) - n_vf} voided)")
    print(f"Parent families: {df['family'].nunique()} | distinct parents: {df['parent_id'].nunique()}")
    print(f"Device: {DEVICE} | N_JOBS: {N_JOBS}")
    print(f"RepeatedKFold({N_SPLITS_OUTER}x{N_REPEATS}) pooled predictions = {len(y) * N_REPEATS}\n")

    rkf = RepeatedKFold(n_splits=N_SPLITS_OUTER, n_repeats=N_REPEATS, random_state=SEED)
    gkf = GroupKFold(n_splits=GKF_SPLITS)

    # Interpolation (RepeatedKFold)
    print(f"INTERPOLATION  RepeatedKFold({N_SPLITS_OUTER}x{N_REPEATS})  [nested tuning per fold]")
    interp = run_cv(rkf, group_tuning=False)
    print(f"  pooled : R2={interp['R2']:.3f}  RMSE={interp['RMSE']:.1f}  MAE={interp['MAE']:.1f}"
          f"  MAPE={interp['MAPE']:.2f}%  (n={interp['n_pred']})")
    print(f"  fold-wise (mean +/- SD): R2={interp['fold_R2_mean']:.3f}+/-{interp['fold_R2_sd']:.3f}"
          f"  RMSE={interp['fold_RMSE_mean']:.1f}+/-{interp['fold_RMSE_sd']:.1f}"
          f"  MAE={interp['fold_MAE_mean']:.1f}+/-{interp['fold_MAE_sd']:.1f}"
          f"  MAPE={interp['fold_MAPE_mean']:.2f}+/-{interp['fold_MAPE_sd']:.2f}\n")

    # Generalisation (GroupKFold over parent microstructures)
    print(f"GENERALISATION  GroupKFold({GKF_SPLITS})  [hold parent microstructures out; group-aware tuning]")
    gen = run_cv(gkf, groups=parent_groups, group_tuning=True)
    print(f"  pooled : R2={gen['R2']:.3f}  RMSE={gen['RMSE']:.1f}  MAE={gen['MAE']:.1f}"
          f"  MAPE={gen['MAPE']:.2f}%  (n={gen['n_pred']})")
    print(f"  fold-wise (mean +/- SD): R2={gen['fold_R2_mean']:.3f}+/-{gen['fold_R2_sd']:.3f}"
          f"  RMSE={gen['fold_RMSE_mean']:.1f}+/-{gen['fold_RMSE_sd']:.1f}"
          f"  MAE={gen['fold_MAE_mean']:.1f}+/-{gen['fold_MAE_sd']:.1f}"
          f"  MAPE={gen['fold_MAPE_mean']:.2f}+/-{gen['fold_MAPE_sd']:.2f}")
    ci = parent_bootstrap_ci(gen["oof"], parent_groups)
    print(f"  parent-bootstrap 95% CI ({N_BOOTSTRAP} resamples): "
          f"R2=[{ci['R2'][0]:.3f}, {ci['R2'][1]:.3f}]  "
          f"RMSE=[{ci['RMSE'][0]:.1f}, {ci['RMSE'][1]:.1f}]  "
          f"MAE=[{ci['MAE'][0]:.1f}, {ci['MAE'][1]:.1f}]  "
          f"MAPE=[{ci['MAPE'][0]:.2f}, {ci['MAPE'][1]:.2f}]\n")

    # Baselines
    print("BASELINES  (identical CV, in-fold scaling, log-target)")
    baselines = {
        "Mean": DummyRegressor(strategy="mean"),
        "GradBoosting": TransformedTargetRegressor(GradientBoostingRegressor(random_state=SEED), func=np.log, inverse_func=np.exp),
    }
    print(f"  {'model':<14}{'Interp R2':>11}{'Gen R2':>9}")
    base_results = {}
    for nm, mdl in baselines.items():
        r = baseline_cv(mdl, rkf); g = baseline_cv(mdl, gkf, groups=parent_groups)
        base_results[nm] = {"interp": r, "gen": g}
        print(f"  {nm:<14}{r['R2']:>11.3f}{g['R2']:>9.3f}")
    print(f"  {'DNN (ours)':<14}{interp['R2']:>11.3f}{gen['R2']:>9.3f}\n")

    # Final model + ablation (full-data fit)
    X, _ = build_Xy(df, "pred"); (Xs,) = scale_col0(X[:, 0], X)
    print("Tuning final model on full data (group-aware) ...")
    HP = tune(X, y, inner_groups=parent_groups)
    print(f"  Final (representative) HP for Table 2: {HP}")
    final = EnsembleDNN(n_models=N_MODELS_FINAL, **HP).fit(Xs, y, groups=parent_groups)
    single = DNNRegressor(**HP).fit(Xs, y, groups=parent_groups)
    print("\nABLATION  (full-data fit R2; TTA disabled by default)")
    print(f"  single DNN     : {r2_score(y, single.predict(Xs, use_tta=False)):.3f}")
    print(f"  ensemble       : {r2_score(y, final.predict(Xs, use_tta=False)):.3f}")
    print(f"Full-data fit R2 (reference only): {r2_score(y, final.predict(Xs, use_tta=False)):.3f}")

    abl = {}
    if CV_ABLATION:
        print("\nABLATION  (cross-validated, RepeatedKFold interpolation, pooled R2)")
        def _abl_cv(n_models, tta_passes):
            hp = dict(HP); hp["tta"] = tta_passes
            yt, yp = [], []
            for tr, te in rkf.split(X):
                Xtr, Xte = scale_col0(X[tr][:, 0], X[tr], X[te])
                e = EnsembleDNN(n_models=n_models, **hp).fit(Xtr, y[tr])
                yt += list(y[te]); yp += list(e.predict(Xte, use_tta=(tta_passes > 1)))
            return r2_score(np.array(yt), np.array(yp))
        abl["single"] = _abl_cv(1, 1)
        abl["ensemble"] = _abl_cv(N_MODELS_CV, 1)
        abl["ensemble_tta"] = _abl_cv(N_MODELS_CV, 30)
        print(f"  single DNN         : {abl['single']:.3f}")
        print(f"  ensemble           : {abl['ensemble']:.3f}")
        print(f"  ensemble + TTA(30) : {abl['ensemble_tta']:.3f}   (TTA shown for comparison only)")
    print()

    # Hierarchical SHAP + controlled analyses
    m1, Xs1, n1 = fit_ensemble(df, "L1", HP)
    g1, _ = shap_analysis(lambda A: m1.predict(A, use_tta=False), Xs1, n1)
    _report("SHAP Level 1  (void presence vs grain descriptors)", g1)

    dgb = df[(df["Void Shape"] != "None") & (df["Void Location"] == "Grain Boundary")].reset_index(drop=True)
    ms, Xss, nms = fit_ensemble(dgb, "shape", HP)
    gshape, sv_shape = shap_analysis(lambda A: ms.predict(A, use_tta=False), Xss, nms)
    _report("SHAP controlled SHAPE  (spherical-GB vs irregular-GB; location fixed)", gshape)

    dsp = df[df["Void Shape"] == "Spherical"].reset_index(drop=True)
    ml, Xsl, nml = fit_ensemble(dsp, "loc", HP)
    gloc, sv_loc = shap_analysis(lambda A: ml.predict(A, use_tta=False), Xsl, nml)
    _report("SHAP controlled LOCATION  (interior vs GB, spherical only; shape fixed)", gloc)

    # SHAP stability + gradient-boosting cross-check
    def _stability(label, predict_fn, Xs_, names_, frame_, level_):
        print(f"\nSHAP stability -- {label}:")
        for s in range(3):
            gb, _ = shap_analysis(predict_fn, Xs_, names_,
                                  nsamples=max(60, SHAP_NSAMPLES // 2), bg_seed=s)
            print(f"  DNN bg-seed {s}: {_order(_named(gb))}")
        Xg, ng = build_Xy(frame_, level_)
        (Xgs,) = scale_col0(Xg[:, 0], Xg)
        gbr = GradientBoostingRegressor(random_state=SEED).fit(
            Xgs, frame_["UHTS (MPa)"].values.astype(np.float32))
        gg = _named(dict(zip(ng, np.abs(shap.TreeExplainer(gbr).shap_values(Xgs)).mean(0))))
        print(f"  GBoost cross-check: {_order(gg)}")

    _stability("Level 1 (void presence)",
               lambda A: m1.predict(A, use_tta=False), Xs1, n1, df, "L1")
    _stability("controlled SHAPE",
               lambda A: ms.predict(A, use_tta=False), Xss, nms, dgb, "shape")
    _stability("controlled LOCATION",
               lambda A: ml.predict(A, use_tta=False), Xsl, nml, dsp, "loc")

    # Figures (bar plots only)
    make_figures(g1, gshape, gloc)
    print("\nSaved figures: shap_level1_bar.png, shap_controlled_bar.png")

    # Summary
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  Interpolation RKF({N_SPLITS_OUTER}x{N_REPEATS}) : R2={interp['R2']:.3f} "
          f"RMSE={interp['RMSE']:.1f} MAPE={interp['MAPE']:.2f}% (n={interp['n_pred']})")
    print(f"  Generalisation GroupKFold({GKF_SPLITS}, parents) : R2={gen['R2']:.3f} "
          f"(fold {gen['fold_R2_mean']:.3f} +/- {gen['fold_R2_sd']:.3f})")
    print(f"  SHAP L1 (presence)     : {_order(g1)}")
    print(f"  SHAP shape (GB voids)  : {_order(gshape)}")
    print(f"  SHAP location (sph.)   : {_order(gloc)}")
    print("=" * 64)

    # Values for the paper
    def _sh(g):
        t = sum(g.values()); return {k: (v, 100 * v / t) for k, v in g.items()}
    print("\n" + "#" * 64)
    print("#  VALUES FOR THE PAPER")
    print("#" * 64)

    print("\n--- TABLE 2  (DNN hyperparameters) ---")
    hp_map = [("Hidden layers (searched)", f"{HP['hidden']}"),
              ("Ensemble size", f"{N_MODELS_FINAL}"),
              ("Dropout rate  (p)", f"{HP['dropout']:.3f}"),
              ("Learning rate (eta)", f"{HP['lr']:.2e}"),
              ("L2 weight decay (lambda)", f"{HP['wd']:.2e}"),
              ("Input noise std (sigma_in)", f"{HP['in_noise']:.3f}"),
              ("Hidden noise std (sigma_hid)", f"{HP['h_noise']:.3f}"),
              ("Early-stopping patience (tau)", f"{HP['patience']} epochs"),
              ("Batch size", f"{HP['bs']}"),
              ("Maximum epochs", f"{HP['epochs']}"),
              ("Activation", "GELU"), ("Optimiser", "Adam"),
              ("LR schedule", "Cosine annealing")]
    for a, b in hp_map:
        print(f"    {a:<30}{b}")

    print("\n--- TABLE 3  (predictive performance) ---")
    print(f"    Interpolation RKF({N_SPLITS_OUTER}x{N_REPEATS}), pooled (n={interp['n_pred']}):")
    print(f"      R2={interp['R2']:.3f}  RMSE={interp['RMSE']:.1f}  MAE={interp['MAE']:.1f}  MAPE={interp['MAPE']:.2f}%")
    print(f"      fold-wise: R2={interp['fold_R2_mean']:.3f}+/-{interp['fold_R2_sd']:.3f}"
          f"  RMSE={interp['fold_RMSE_mean']:.1f}+/-{interp['fold_RMSE_sd']:.1f}"
          f"  MAE={interp['fold_MAE_mean']:.1f}+/-{interp['fold_MAE_sd']:.1f}"
          f"  MAPE={interp['fold_MAPE_mean']:.2f}+/-{interp['fold_MAPE_sd']:.2f}")
    print(f"    Generalisation GroupKFold({GKF_SPLITS}, parents), pooled (n={gen['n_pred']}):")
    print(f"      R2={gen['R2']:.3f}  RMSE={gen['RMSE']:.1f}  MAE={gen['MAE']:.1f}  MAPE={gen['MAPE']:.2f}%")
    print(f"      fold-wise: R2={gen['fold_R2_mean']:.3f}+/-{gen['fold_R2_sd']:.3f}"
          f"  RMSE={gen['fold_RMSE_mean']:.1f}+/-{gen['fold_RMSE_sd']:.1f}"
          f"  MAE={gen['fold_MAE_mean']:.1f}+/-{gen['fold_MAE_sd']:.1f}"
          f"  MAPE={gen['fold_MAPE_mean']:.2f}+/-{gen['fold_MAPE_sd']:.2f}")
    print(f"      95% CI (parent bootstrap): R2=[{ci['R2'][0]:.3f}, {ci['R2'][1]:.3f}]"
          f"  RMSE=[{ci['RMSE'][0]:.1f}, {ci['RMSE'][1]:.1f}]"
          f"  MAE=[{ci['MAE'][0]:.1f}, {ci['MAE'][1]:.1f}]"
          f"  MAPE=[{ci['MAPE'][0]:.2f}, {ci['MAPE'][1]:.2f}]")
    print("    Baselines (R2, interpolation / generalisation):")
    for nm in ["GradBoosting", "Mean"]:
        print(f"      {nm:<14}{base_results[nm]['interp']['R2']:>7.3f} / {base_results[nm]['gen']['R2']:>6.3f}")
    print(f"      {'DNN (ours)':<14}{interp['R2']:>7.3f} / {gen['R2']:>6.3f}")
    if abl:
        print("    Ablation (cross-validated interpolation R2):")
        print(f"      single DNN {abl['single']:.3f} | ensemble {abl['ensemble']:.3f} | "
              f"ensemble+TTA {abl['ensemble_tta']:.3f}")

    print("\n--- SHAP RESULTS (mean |SHAP| in MPa, and share %) ---")
    for label, g in [("Level 1 (presence vs grain)", g1),
                     ("Controlled shape (GB voids)", gshape),
                     ("Controlled location (spherical)", gloc)]:
        print(f"    {label}:")
        for k, (v, p) in sorted(_sh(g).items(), key=lambda kv: -kv[1][0]):
            print(f"      {k:<18}{v:>7.1f} MPa   {p:5.1f}%")
    print("#" * 64)


if __name__ == "__main__":
    main()
