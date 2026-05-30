# ================================================================
#  DNN Surrogate Model for UHTS Prediction of Fe80Cr16Ni4
#  Nanocrystals with Grain Morphology and Nanovoid Descriptors
#
#  Authors: Mahshad Fani, Oluwatimilehin Akinloye,
#           Yingtao Liu, Shuozhi Xu
#  University of Oklahoma, 2026
# ================================================================

# ================================================================
#  1. Dependencies
# ================================================================
import subprocess, sys
for pkg in ["optuna", "shap", "torch"]:
    try: __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", pkg, "-q"])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rcParams
import shap, warnings, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, KFold, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error)

rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.linewidth'] = 0.8
SEED = 42
TTA_PASSES = 30
torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================================================
#  2. Dataset
#  108 atomistic simulations:
#    - 18 void-free (3 replicates × 2 morphologies × 3 grain sizes)
#    - 90 void-containing (5 replicates × 6 configurations × 3 grain sizes)
# ================================================================
rows = []

for gs, shape, vals in [
    (10, 'Equiaxed', [4883.738, 4910.627, 5000.630]),
    (15, 'Equiaxed', [5658.105, 5563.383, 5686.377]),
    (20, 'Equiaxed', [4743.231, 4890.020, 4761.052]),
    (10, 'Columnar', [5195.072, 5349.290, 5612.630]),
    (15, 'Columnar', [5559.355, 5316.518, 5424.627]),
    (20, 'Columnar', [4516.982, 4790.460, 4980.750]),
]:
    for v in vals:
        rows.append((gs, shape, 'None', 'None', v))

for gs, shape, vals in [
    (10, 'Equiaxed', [3503.188, 3354.142, 3419.141, 3368.555, 3390.060]),
    (15, 'Equiaxed', [3959.253, 4022.380, 4110.265, 3939.886, 3960.746]),
    (20, 'Equiaxed', [3585.079, 3640.643, 3566.088, 3568.024, 3495.653]),
    (10, 'Columnar', [4197.749, 4129.078, 4567.889, 4331.483, 4298.854]),
    (15, 'Columnar', [4374.370, 4119.882, 4028.399, 4357.496, 4357.496]),
    (20, 'Columnar', [3822.848, 3859.674, 3834.406, 3603.685, 3473.315]),
]:
    for v in vals:
        rows.append((gs, shape, 'Spherical', 'Interior', v))

for gs, shape, vals in [
    (10, 'Equiaxed', [3245.698, 3367.360, 3390.535, 3474.526, 3342.819]),
    (15, 'Equiaxed', [4110.265, 3993.455, 4112.391, 3922.710, 3991.429]),
    (20, 'Equiaxed', [3585.079, 3734.433, 3679.311, 3516.168, 3476.085]),
    (10, 'Columnar', [4147.648, 4288.204, 3837.379, 4140.089, 4233.086]),
    (15, 'Columnar', [4219.584, 3999.552, 4010.205, 3859.290, 3860.842]),
    (20, 'Columnar', [3473.315, 3800.727, 3274.761, 3406.695, 3402.717]),
]:
    for v in vals:
        rows.append((gs, shape, 'Spherical', 'Grain Boundary', v))

for gs, shape, vals in [
    (10, 'Equiaxed', [3269.777, 3147.577, 3340.144, 3340.144, 3273.410]),
    (15, 'Equiaxed', [3822.103, 3959.374, 3860.706, 3956.963, 3836.096]),
    (20, 'Equiaxed', [3581.370, 3406.695, 3402.717, 3516.168, 3353.806]),
    (10, 'Columnar', [4164.610, 4150.085, 4120.846, 3998.664, 4328.446]),
    (15, 'Columnar', [3830.150, 3859.680, 3727.930, 3842.324, 3965.279]),
    (20, 'Columnar', [3676.162, 3646.863, 3015.406, 3392.151, 3338.219]),
]:
    for v in vals:
        rows.append((gs, shape, 'Irregular', 'Grain Boundary', v))

df = pd.DataFrame(rows,
    columns=['Grain Size (nm)', 'Grain Shape',
             'Void Shape', 'Void Location', 'UHTS (MPa)'])

df['Void Presence']     = (df['Void Shape'] != 'None').astype(float)
df['Void Shape Cat']    = df['Void Shape'].replace('None', 'No_Void')
df['Void Location Cat'] = df['Void Location'].replace('None', 'No_Void')
df['group_id'] = (df['Grain Shape'] + '_' +
                  df['Grain Size (nm)'].astype(str) + '_' +
                  df['Void Shape'] + '_' + df['Void Location'])

groups = df['group_id'].values
X = df[['Grain Size (nm)', 'Void Presence', 'Grain Shape',
        'Void Shape Cat', 'Void Location Cat']].copy()
y = df['UHTS (MPa)'].values.astype(np.float32)

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc',  StandardScaler())]),
     ['Grain Size (nm)', 'Void Presence']),
    ('gs', OneHotEncoder(categories=[['Columnar', 'Equiaxed']],
        drop='first', sparse_output=False, handle_unknown='ignore'),
     ['Grain Shape']),
    ('vs', OneHotEncoder(categories=[['No_Void', 'Irregular', 'Spherical']],
        drop='first', sparse_output=False, handle_unknown='ignore'),
     ['Void Shape Cat']),
    ('vl', OneHotEncoder(categories=[['No_Void', 'Grain Boundary', 'Interior']],
        drop='first', sparse_output=False, handle_unknown='ignore'),
     ['Void Location Cat']),
], remainder='drop')

preprocessor.fit(X)
X_proc = preprocessor.transform(X).astype(np.float32)

feature_names = ['Grain Size (nm)', 'Void Presence',
                 'Grain Shape_Equiaxed',
                 'Void Shape_Irregular', 'Void Shape_Spherical',
                 'Void Location_GB', 'Void Location_Interior']

print(f"Dataset: {len(y)} samples "
      f"({(df['Void Shape']=='None').sum()} void-free + "
      f"{(df['Void Shape']!='None').sum()} void-containing)")
print(f"Features: {feature_names}")


# ================================================================
#  3. Model Architecture
#  Residual DNN with 8 regularisation mechanisms:
#  skip connections, input noise, hidden noise, BatchNorm,
#  Dropout, L2 weight decay, early stopping, cosine LR schedule
# ================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_s, out_s, dropout, hn=0.01):
        super().__init__()
        self.hn     = hn
        self.linear = nn.Linear(in_s, out_s)
        self.bn     = nn.BatchNorm1d(out_s)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(dropout)
        self.skip   = (nn.Linear(in_s, out_s, bias=False)
                       if in_s != out_s else nn.Identity())

    def forward(self, x):
        out = self.bn(self.linear(x))
        if self.training and self.hn > 0:
            out = out + torch.randn_like(out) * self.hn
        return self.act(self.drop(out)) + self.skip(x)


class DNN(nn.Module):
    def __init__(self, n_in, hidden, dropout, hn=0.01):
        super().__init__()
        blocks, in_s = [], n_in
        for h in hidden:
            blocks.append(ResidualBlock(in_s, h, dropout, hn))
            in_s = h
        self.blocks = nn.ModuleList(blocks)
        self.head   = nn.Linear(in_s, 1)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(-1)


class DNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_sizes=(64, 64), dropout=0.2,
                 lr=1e-3, weight_decay=1e-4, batch_size=16,
                 n_epochs=600, patience=40,
                 input_noise_std=0.01, hidden_noise_std=0.01,
                 tta_passes=TTA_PASSES):
        self.hidden_sizes     = hidden_sizes
        self.dropout          = dropout
        self.lr               = lr
        self.weight_decay     = weight_decay
        self.batch_size       = batch_size
        self.n_epochs         = n_epochs
        self.patience         = patience
        self.input_noise_std  = input_noise_std
        self.hidden_noise_std = hidden_noise_std
        self.tta_passes       = tta_passes

    def _scale_y(self, y):
        self.ym = float(y.mean())
        self.ys = float(y.std()) if y.std() > 0 else 1.0
        return (y - self.ym) / self.ys

    def _unscale_y(self, ys):
        return ys * self.ys + self.ym

    def fit(self, X, y):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(self._scale_y(y), dtype=torch.float32)
        nv = max(1, int(len(Xt) * 0.15))
        idx = np.random.permutation(len(Xt))
        vi, ti = idx[:nv], idx[nv:]
        self.model_ = DNN(Xt.shape[1], self.hidden_sizes,
                          self.dropout, self.hidden_noise_std)
        opt = torch.optim.Adam(self.model_.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.n_epochs)
        lf  = nn.MSELoss()
        bs  = min(self.batch_size, max(2, len(ti) // 4))
        ld  = DataLoader(TensorDataset(Xt[ti], yt[ti]),
                         batch_size=bs, shuffle=True, drop_last=True)
        best_val, best_weights, wait = float('inf'), None, 0
        for _ in range(self.n_epochs):
            self.model_.train()
            for xb, yb in ld:
                xn = xb + torch.randn_like(xb) * self.input_noise_std
                opt.zero_grad()
                lf(self.model_(xn), yb).backward()
                opt.step()
            sch.step()
            self.model_.eval()
            with torch.no_grad():
                vl = lf(self.model_(Xt[vi]), yt[vi]).item()
            if vl < best_val - 1e-5:
                best_val = vl
                best_weights = {k: v.clone() for k, v in
                                self.model_.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_weights:
            self.model_.load_state_dict(best_weights)
        self.model_.eval()
        return self

    def predict(self, X, use_tta=True):
        self.model_.eval()
        Xt = torch.tensor(X, dtype=torch.float32)
        if use_tta and self.tta_passes > 1:
            preds = []
            with torch.no_grad():
                for _ in range(self.tta_passes):
                    xn = Xt + torch.randn_like(Xt) * self.input_noise_std
                    preds.append(self.model_(xn).numpy())
            ys = np.stack(preds).mean(axis=0)
        else:
            with torch.no_grad():
                ys = self.model_(Xt).numpy()
        return self._unscale_y(ys)


class EnsembleDNN(BaseEstimator, RegressorMixin):
    def __init__(self, n_models=20, **kw):
        self.n_models = n_models
        self.kw = kw

    def fit(self, X, y):
        self.models_ = []
        for s in range(self.n_models):
            torch.manual_seed(s)
            np.random.seed(s)
            m = DNNRegressor(**self.kw)
            m.fit(X, y)
            self.models_.append(m)
        return self

    def predict(self, X, use_tta=True):
        return np.stack([m.predict(X, use_tta=use_tta)
                         for m in self.models_]).mean(axis=0)


# ================================================================
#  4. Hyperparameter Optimisation
#  Tree-structured Parzen Estimator via Optuna
#  20 trials × 5-fold cross-validation
#  Architecture fixed at (64, 64)
# ================================================================
ARCH = (64, 64)
cvo  = KFold(n_splits=5, shuffle=True, random_state=SEED)

def objective(trial):
    kw = dict(
        hidden_sizes     = ARCH,
        dropout          = trial.suggest_float('dropout',
                                               0.10, 0.40),
        lr               = trial.suggest_float('lr',
                                               1e-4, 5e-3, log=True),
        weight_decay     = trial.suggest_float('weight_decay',
                                               1e-5, 1e-2, log=True),
        input_noise_std  = trial.suggest_float('input_noise_std',
                                               0.005, 0.05),
        hidden_noise_std = trial.suggest_float('hidden_noise_std',
                                               0.005, 0.05),
        patience         = trial.suggest_int('patience', 25, 60),
        n_epochs=600, batch_size=16, tta_passes=5)
    scores = []
    for tr, val in cvo.split(X_proc):
        ens = EnsembleDNN(n_models=5, **kw)
        ens.fit(X_proc[tr], y[tr])
        scores.append(r2_score(y[val], ens.predict(X_proc[val])))
    return float(np.mean(scores))

study = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=20, show_progress_bar=True)

bp = study.best_params
HP = dict(
    hidden_sizes     = ARCH,
    dropout          = bp['dropout'],
    lr               = bp['lr'],
    weight_decay     = bp['weight_decay'],
    input_noise_std  = bp['input_noise_std'],
    hidden_noise_std = bp['hidden_noise_std'],
    patience         = bp['patience'],
    n_epochs=600, batch_size=16, tta_passes=TTA_PASSES)

print(f"\nOptuna best CV R² = {study.best_value:.3f}")
print(f"Optimised hyperparameters:")
print(f"  dropout          = {HP['dropout']:.3f}")
print(f"  lr               = {HP['lr']:.5f}")
print(f"  weight_decay     = {HP['weight_decay']:.6f}")
print(f"  input_noise_std  = {HP['input_noise_std']:.4f}")
print(f"  hidden_noise_std = {HP['hidden_noise_std']:.4f}")
print(f"  patience         = {HP['patience']}")


# ================================================================
#  5. Cross-Validation
#
#  Primary:   RepeatedKFold(10×5) — pooled R² over 510 predictions
#  Secondary: GroupKFold(6)       — all replicates of each
#                                   configuration held out together
# ================================================================
print("\nPRIMARY: RepeatedKFold(10×5)\n")
rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=SEED)
r2s, rkf_rmse, rkf_mae = [], [], []
yt_all, yp_all = [], []

for fold, (tr, val) in enumerate(rkf.split(X_proc)):
    ens = EnsembleDNN(n_models=20, **HP)
    ens.fit(X_proc[tr], y[tr])
    preds = ens.predict(X_proc[val])
    r2s.append(r2_score(y[val], preds))
    rkf_rmse.append(np.sqrt(mean_squared_error(y[val], preds)))
    rkf_mae.append(mean_absolute_error(y[val], preds))
    yt_all.extend(y[val])
    yp_all.extend(preds)
    if (fold + 1) % 10 == 0:
        print(f"  Fold {fold+1}/50  R²={np.mean(r2s):.3f} "
              f"± {np.std(r2s):.3f}")

r2s    = np.array(r2s)
yt_all = np.array(yt_all)
yp_all = np.array(yp_all)
pr2    = r2_score(yt_all, yp_all)
prmse  = np.sqrt(mean_squared_error(yt_all, yp_all))
pmae   = mean_absolute_error(yt_all, yp_all)
pmape  = np.mean(np.abs((yt_all - yp_all) / yt_all)) * 100

print(f"\nPRIMARY RKF(10×5) Results:")
print(f"  Pooled R²  = {pr2:.3f}")
print(f"  Mean R²    = {r2s.mean():.3f} ± {r2s.std():.3f}")
print(f"  RMSE       = {prmse:.1f} MPa")
print(f"  MAE        = {pmae:.1f} MPa")
print(f"  MAPE       = {pmape:.2f}%")

print("\nSECONDARY: GroupKFold(6)\n")
gkf = GroupKFold(n_splits=6)
gr2s, gyt, gyp = [], [], []

for fold, (tr, val) in enumerate(gkf.split(X_proc, y, groups)):
    ens = EnsembleDNN(n_models=20, **HP)
    ens.fit(X_proc[tr], y[tr])
    preds = ens.predict(X_proc[val])
    gr2s.append(r2_score(y[val], preds))
    gyt.extend(y[val])
    gyp.extend(preds)
    print(f"  Fold {fold+1}/6  R²={gr2s[-1]:.3f}")

gyt    = np.array(gyt)
gyp    = np.array(gyp)
gpr2   = r2_score(gyt, gyp)
gprmse = np.sqrt(mean_squared_error(gyt, gyp))

print(f"\nSECONDARY GKF(6) Results:")
print(f"  Pooled R² = {gpr2:.3f}")
print(f"  RMSE      = {gprmse:.1f} MPa")


# ================================================================
#  6. Final Ensemble (20 models, full dataset)
# ================================================================
torch.manual_seed(SEED)
np.random.seed(SEED)
final_ens = EnsembleDNN(n_models=20, **HP)
final_ens.fit(X_proc, y)

full_r2_tta   = r2_score(y, final_ens.predict(X_proc, use_tta=True))
full_r2_notta = r2_score(y, final_ens.predict(X_proc, use_tta=False))

print(f"\nFinal Ensemble Results:")
print(f"  Full-data R² (with TTA)    = {full_r2_tta:.3f}")
print(f"  Full-data R² (without TTA) = {full_r2_notta:.3f}")
print(f"  Overfitting gap            = {full_r2_tta - pr2:.3f}")


# ================================================================
#  7. SHAP Analysis
#  KernelExplainer with 10-sample k-means background
#  200 perturbation samples per evaluation
#  TTA disabled for deterministic attribution values
# ================================================================
def predict_no_tta(X):
    return final_ens.predict(X, use_tta=False)

bg  = shap.kmeans(X_proc, 10)
exp = shap.KernelExplainer(predict_no_tta, bg)
sv  = exp.shap_values(X_proc, nsamples=200)
ir  = np.abs(sv).mean(0)

def aggregate_shap(feature_names, raw_importance):
    out = {'Void Presence': 0., 'Grain Size': 0., 'Grain Shape': 0.,
           'Void Shape': 0., 'Void Location': 0.}
    for f, v in zip(feature_names, raw_importance):
        if   f == 'Void Presence':    out['Void Presence'] += v
        elif f == 'Grain Size (nm)':  out['Grain Size']    += v
        elif 'Grain Shape'   in f:    out['Grain Shape']   += v
        elif 'Void Shape'    in f:    out['Void Shape']    += v
        elif 'Void Location' in f:    out['Void Location'] += v
    return out

imp    = aggregate_shap(feature_names, ir)
ranked = sorted(imp.keys(), key=lambda k: -imp[k])
total  = sum(imp.values())

print("\nSHAP Ranking:")
print("  " + " > ".join(ranked))
print(f"\n  {'Feature':<20} {'SHAP (MPa)':>12}  {'%':>6}")
print(f"  {'-'*40}")
for feat in ranked:
    print(f"  {feat:<20} {imp[feat]:>10.1f}   "
          f"{imp[feat]/total*100:>5.1f}%")


# ================================================================
#  8. Figures
# ================================================================

# Figure 1: SHAP Bar Chart
fo   = sorted(imp.keys(), key=lambda f: imp[f])
vp   = [imp[f] for f in fo]
clrs = [('#d62728' if f in ['Void Presence', 'Void Shape',
                             'Void Location']
         else '#1f77b4') for f in fo]

fig1, ax1 = plt.subplots(figsize=(7.2, 3.8), dpi=300)
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')
bars = ax1.barh(range(len(fo)), vp, height=0.55, color=clrs,
                alpha=0.90, edgecolor='black', linewidth=0.35,
                zorder=3)
mx = max(vp)
for bar, v in zip(bars, vp):
    ax1.text(bar.get_width() + mx * 0.018,
             bar.get_y() + bar.get_height() / 2,
             f'{v:.1f}', va='center', ha='left',
             fontsize=10, color='black')
ax1.set_yticks(range(len(fo)))
ax1.set_yticklabels(fo, fontsize=11, color='black')
ax1.set_xlabel('Mean |SHAP value| (MPa)', fontsize=12, labelpad=8)
ax1.set_xlim(0, mx * 1.25)
ax1.tick_params(axis='x', labelsize=10, colors='black')
ax1.tick_params(axis='y', length=0)
ax1.xaxis.grid(True, linestyle='--', linewidth=0.45,
               color='0.82', zorder=1)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_linewidth(0.8)
ax1.spines['bottom'].set_color('black')
ax1.legend(handles=[
    Patch(facecolor='#1f77b4', alpha=0.90, label='Grain descriptors'),
    Patch(facecolor='#d62728', alpha=0.90, label='Void descriptors')],
    fontsize=9.5, frameon=True, framealpha=1.0,
    edgecolor='0.75', loc='lower right')
plt.tight_layout()
plt.savefig('SHAP_bar.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.show()


# Figure 2: SHAP Beeswarm
Xdf = pd.DataFrame(X_proc, columns=feature_names)
plt.figure(figsize=(7.2, 4.2), dpi=300)
shap.summary_plot(sv, Xdf, show=False, color_bar=True,
                  plot_size=None, max_display=len(feature_names))
ax = plt.gca()
ax.set_facecolor('white')
plt.gcf().set_facecolor('white')
ax.set_xlabel('SHAP value (impact on model output, MPa)',
              fontsize=12, labelpad=8)
ax.tick_params(axis='both', labelsize=10, colors='black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)
plt.tight_layout()
plt.savefig('SHAP_beeswarm.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.show()


# ================================================================
#  9. Summary
# ================================================================
print("\n" + "="*60)
print("  PUBLICATION SUMMARY")
print("="*60)
print(f"\n  PRIMARY — RKF(10×5)")
print(f"    Pooled R²  : {pr2:.3f}")
print(f"    Mean R²    : {r2s.mean():.3f} ± {r2s.std():.3f}")
print(f"    RMSE       : {prmse:.1f} MPa")
print(f"    MAE        : {pmae:.1f} MPa")
print(f"    MAPE       : {pmape:.2f}%")
print(f"\n  SECONDARY — GKF(6)")
print(f"    Pooled R²  : {gpr2:.3f}")
print(f"    RMSE       : {gprmse:.1f} MPa")
print(f"\n  Full-data R²   : {full_r2_tta:.3f}")
print(f"  Overfitting gap: {full_r2_tta - pr2:.3f}")
print(f"\n  SHAP Ranking:")
for i, feat in enumerate(ranked, 1):
    print(f"    {i}. {feat:<20} {imp[feat]:>6.1f} MPa  "
          f"({imp[feat]/total*100:.1f}%)")
print("="*60)