"""
Physics-Hybrid SVR for AWJM - Final Model
==========================================
Architecture (same as ANN hybrid, with SVR residual):

  STEP 1 - Physics Baseline (fitted from data):
    Depth of Cut : h = K1 * d^alpha * v^beta * SOD^gamma * H^delta
    Kerf Width   : W = K2 * d  + K3 * SOD

  STEP 2 - SVR Residual:
    Two separate SVR models learn the deviation (actual - physics_baseline)
    One for KW residual, one for DoC residual
    Inputs: [H, d, v, SOD]  (scaled)

  STEP 3 - Final Prediction:
    output = physics_baseline + SVR_residual
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

FEATURES   = ["hardness", "nozzle_dia", "traverse_speed", "SOD"]
TARGETS    = ["kerf_width", "depth_of_cut"]
LABELS     = {"kerf_width": "Kerf Width (mm)", "depth_of_cut": "Depth of Cut (mm)"}
COLORS     = {"kerf_width": "#6C63FF", "depth_of_cut": "#FF6584"}
MAT_COLORS = {"Titanium": "#6C63FF", "Inconel": "#FF6584", "SS316": "#43B89C"}
MAT_HV     = {"Titanium": 385, "Inconel": 485, "SS316": 425}
PREFIX     = "svr"   # file prefix for all outputs

# ======================================================================
# 1.  DATA PARSERS  (identical to ANN hybrid)
# ======================================================================
def parse_ti(path):
    raw = pd.read_excel(path, sheet_name="Titanium Experiments ", header=None)
    rows = []
    for nz, cols in [(1.57, [0,1,2,3]), (0.76, [6,7,8,9])]:
        blk = raw.iloc[3:, cols].copy()
        blk.columns = ["traverse_speed","SOD","depth_of_cut","kerf_width"]
        blk["traverse_speed"] = blk["traverse_speed"].ffill()
        blk = blk.apply(pd.to_numeric, errors="coerce").dropna(
            subset=["SOD","depth_of_cut","kerf_width"])
        blk["nozzle_dia"] = nz; blk["hardness"] = 385; blk["material"] = "Titanium"
        rows.append(blk)
    return pd.concat(rows, ignore_index=True)

def parse_inconel(path):
    raw = pd.read_excel(path, sheet_name="Sheet1", header=None)
    rows = []
    for nz, cols in [(1.57, [0,1,2,3,4]), (0.76, [9,10,11,12,13])]:
        blk = raw.iloc[2:, cols].copy()
        blk.columns = ["scan_no","traverse_speed","SOD","kerf_width","depth_of_cut"]
        blk["traverse_speed"] = blk["traverse_speed"].ffill()
        blk = blk.apply(pd.to_numeric, errors="coerce").dropna(
            subset=["SOD","depth_of_cut","kerf_width"])
        blk["nozzle_dia"] = nz; blk["hardness"] = 485; blk["material"] = "Inconel"
        rows.append(blk)
    return pd.concat(rows, ignore_index=True)

def parse_ss(path):
    raw = pd.read_excel(path, sheet_name="Experiment on Nozzle 0.3556", header=None)
    rows = []
    for nz, cols in [(0.76, [0,1,2,3]), (1.57, [9,10,11,12])]:
        blk = raw.iloc[3:, cols].copy()
        blk.columns = ["traverse_speed","SOD","depth_of_cut","kerf_width"]
        blk["traverse_speed"] = blk["traverse_speed"].ffill()
        blk = blk.apply(pd.to_numeric, errors="coerce").dropna(
            subset=["SOD","depth_of_cut","kerf_width"])
        blk["nozzle_dia"] = nz; blk["hardness"] = 425; blk["material"] = "SS316"
        rows.append(blk)
    return pd.concat(rows, ignore_index=True)

# ======================================================================
# 2.  PHYSICS MODEL  (identical to ANN hybrid)
# ======================================================================
def fit_physics_constants(df_train):
    h  = df_train["depth_of_cut"].values
    W  = df_train["kerf_width"].values
    d  = df_train["nozzle_dia"].values
    v  = df_train["traverse_speed"].values
    s  = df_train["SOD"].values
    H  = df_train["hardness"].values

    log_h = np.log(h)
    A_doc = np.column_stack([np.ones(len(h)), np.log(d), np.log(v), np.log(s), np.log(H)])
    coeffs_doc, _, _, _ = np.linalg.lstsq(A_doc, log_h, rcond=None)
    log_K1, alpha, beta, gamma, delta = coeffs_doc
    K1 = np.exp(log_K1)

    A_kw = np.column_stack([d, s])
    K2, K3 = np.linalg.lstsq(A_kw, W, rcond=None)[0]

    return dict(K1=float(K1), alpha=float(alpha), beta=float(beta),
                gamma=float(gamma), delta=float(delta),
                K2=float(K2), K3=float(K3))

def physics_baseline(df, p):
    d = df["nozzle_dia"].values; v = df["traverse_speed"].values
    s = df["SOD"].values;        H = df["hardness"].values
    log_h = (np.log(p["K1"]) + p["alpha"]*np.log(d) + p["beta"]*np.log(v)
             + p["gamma"]*np.log(s) + p["delta"]*np.log(H))
    return np.column_stack([p["K2"]*d + p["K3"]*s, np.exp(log_h)])

def physics_baseline_raw(H, d, v, SOD, p):
    log_h = (np.log(p["K1"]) + p["alpha"]*np.log(d) + p["beta"]*np.log(v)
             + p["gamma"]*np.log(SOD) + p["delta"]*np.log(H))
    return np.stack([p["K2"]*d + p["K3"]*SOD, np.exp(log_h)], axis=-1)

# ======================================================================
# 3.  BUILD COMBINED DATASET
# ======================================================================
print("=" * 62)
print("  Physics-Hybrid SVR -- Final Model")
print("  DoC : h = K1 * d^a * v^b * SOD^g * H^d")
print("  KW  : W = K2 * d  + K3 * SOD")
print("  Final = Physics Baseline + SVR Residual")
print("=" * 62)

df = pd.concat([
    parse_ti("Ti_nozzle 0.76_and_1.5mm.xlsx"),
    parse_inconel("IN_ nozzle 0.76_and_1.5mm.xlsx"),
    parse_ss("SS_ nozzle 0.76_and_1.5mm.xlsx"),
], ignore_index=True)
df = df[FEATURES + TARGETS + ["material"]].dropna()
print(f"\n[OK] Dataset: {len(df)} samples")

X_raw  = df[FEATURES].values.astype(np.float32)
y_true = df[TARGETS].values.astype(np.float32)
groups = df["material"] + "_" + df["nozzle_dia"].astype(str)

(X_tr, X_te, y_tr, y_te,
 df_tr_idx, df_te_idx,
 g_tr, g_te) = train_test_split(
    X_raw, y_true, df.index, groups,
    test_size=0.20, random_state=42, stratify=groups)

df_train = df.loc[df_tr_idx].copy()
df_test  = df.loc[df_te_idx].copy()
print(f"[OK] Train: {len(X_tr)} | Test: {len(X_te)}")

# ======================================================================
# 4.  FIT PHYSICS CONSTANTS
# ======================================================================
p = fit_physics_constants(df_train)
K1=p["K1"]; K2=p["K2"]; K3=p["K3"]
print(f"\n[OK] Physics power-law fitted (optimal exponents):")
print(f"     DoC : h = {K1:.4e} * d^{p['alpha']:.3f} * v^{p['beta']:.3f} * SOD^{p['gamma']:.3f} * H^{p['delta']:.3f}")
print(f"     KW  : W = {K2:.4f}*d + {K3:.4f}*SOD")

with open(f"{PREFIX}_physics_constants.json", "w") as f:
    json.dump(p, f, indent=2)

phys_tr = physics_baseline(df_train, p).astype(np.float32)
phys_te = physics_baseline(df_test,  p).astype(np.float32)
resid_tr = y_tr - phys_tr
resid_te = y_te - phys_te

print(f"\n{'-'*62}\nPHYSICS-ONLY BASELINE (no ML yet)\n{'-'*62}")
for i, col in enumerate(TARGETS):
    r2   = r2_score(y_te[:, i], phys_te[:, i])
    rmse = np.sqrt(mean_squared_error(y_te[:, i], phys_te[:, i]))
    print(f"  {col:16s}  R2={r2:.4f}  RMSE={rmse:.4f}mm")

print(f"\n  Residual stats (train):")
for i, col in enumerate(TARGETS):
    print(f"  {col:16s}  mean={resid_tr[:,i].mean():.4f}  "
          f"std={resid_tr[:,i].std():.4f}  max_abs={np.abs(resid_tr[:,i]).max():.4f}")

# ======================================================================
# 5.  SCALE INPUTS & RESIDUALS
# ======================================================================
scaler_X = StandardScaler()
scaler_r = StandardScaler()

X_tr_s = scaler_X.fit_transform(X_tr)
X_te_s = scaler_X.transform(X_te)
r_tr_s = scaler_r.fit_transform(resid_tr)
r_te_s = scaler_r.transform(resid_te)

joblib.dump(scaler_X, f"{PREFIX}_scaler_X.pkl")
joblib.dump(scaler_r, f"{PREFIX}_scaler_r.pkl")
print("[OK] Scalers saved.")

# ======================================================================
# 6.  SVR RESIDUAL MODEL  (with GridSearchCV hyperparameter tuning)
# ======================================================================
print("\nTuning SVR hyperparameters (GridSearchCV, 5-fold) ...")

param_grid = {
    "estimator__C":       [1, 10, 100, 500],
    "estimator__gamma":   ["scale", 0.1, 0.5],
    "estimator__epsilon": [0.01, 0.05, 0.1],
}

base_svr = MultiOutputRegressor(SVR(kernel="rbf"))
gs = GridSearchCV(base_svr, param_grid, cv=5, scoring="r2",
                  n_jobs=-1, verbose=0, refit=True)
gs.fit(X_tr_s, r_tr_s)

svr_model = gs.best_estimator_
print(f"[OK] Best SVR params: {gs.best_params_}")
print(f"[OK] Best CV R2:      {gs.best_score_:.4f}")

joblib.dump(svr_model, f"{PREFIX}_residual_model.pkl")
print(f"[OK] Model saved: {PREFIX}_residual_model.pkl")

# ======================================================================
# 7.  FULL HYBRID PREDICTION
# ======================================================================
def hybrid_predict_df(df_in):
    X_raw_in = df_in[FEATURES].values.astype(np.float32)
    phys     = physics_baseline(df_in, p)
    X_s      = scaler_X.transform(X_raw_in)
    resid_s  = svr_model.predict(X_s)
    resid    = scaler_r.inverse_transform(resid_s)
    return phys + resid

def hybrid_predict_raw(H, d, v, SOD):
    H = np.atleast_1d(H); d = np.atleast_1d(d)
    v = np.atleast_1d(v); SOD = np.atleast_1d(SOD)
    phys = physics_baseline_raw(H, d, v, SOD, p)
    X_raw_in = np.column_stack([H, d, v, SOD]).astype(np.float32)
    X_s      = scaler_X.transform(X_raw_in)
    resid_s  = svr_model.predict(X_s)
    resid    = scaler_r.inverse_transform(resid_s)
    return phys + resid

# ======================================================================
# 8.  EVALUATE
# ======================================================================
def evaluate(y_true_arr, y_pred_arr, tag):
    mets = {}
    for i, col in enumerate(TARGETS):
        r2   = r2_score(y_true_arr[:, i], y_pred_arr[:, i])
        rmse = np.sqrt(mean_squared_error(y_true_arr[:, i], y_pred_arr[:, i]))
        mae  = mean_absolute_error(y_true_arr[:, i], y_pred_arr[:, i])
        mape = np.mean(np.abs((y_true_arr[:, i] - y_pred_arr[:, i])
                              / (y_true_arr[:, i] + 1e-9))) * 100
        mets[col] = dict(R2=r2, RMSE=rmse, MAE=mae, MAPE=mape)
        print(f"  [{tag}] {col:16s}  R2={r2:.4f}  RMSE={rmse:.4f}mm  "
              f"MAE={mae:.4f}mm  MAPE={mape:.2f}%")
    return mets

yp_train = hybrid_predict_df(df_train)
yp_test  = hybrid_predict_df(df_test)

print(f"\n{'-'*62}\nFINAL HYBRID SVR MODEL EVALUATION\n{'-'*62}")
mt = evaluate(y_tr, yp_train, "TRAIN")
me = evaluate(y_te, yp_test,  "TEST ")

print(f"\n{'-'*62}\nPER-MATERIAL TEST METRICS\n{'-'*62}")
mat_labels = g_te.str.split("_").str[0].values
for mat in ["Titanium", "Inconel", "SS316"]:
    mask = mat_labels == mat
    if not mask.sum(): continue
    yt_m = y_te[mask]; yp_m = yp_test[mask]
    for i, col in enumerate(TARGETS):
        r2   = r2_score(yt_m[:, i], yp_m[:, i])
        rmse = np.sqrt(mean_squared_error(yt_m[:, i], yp_m[:, i]))
        print(f"  {mat:10s} | {col:16s}  R2={r2:.4f}  RMSE={rmse:.4f}mm  n={mask.sum()}")

# ======================================================================
# 9.  PLOTS
# ======================================================================
plt.style.use("seaborn-v0_8-whitegrid")
N = 300
TAG = "Physics-Hybrid SVR"

# ── Fig 1: Parity plots (physics vs hybrid) ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle(f"Parity Plots -- Physics Baseline vs Final {TAG}\n(Test Set)",
             fontsize=13, fontweight="bold")

for row, col in enumerate(TARGETS):
    for ax, yp_arr, label in [
        (axes[row][0], phys_te[:, row], "Physics Baseline only"),
        (axes[row][1], yp_test[:, row], f"Physics + SVR Residual"),
    ]:
        for mat, mc in MAT_COLORS.items():
            mask = mat_labels == mat
            if mask.sum():
                ax.scatter(y_te[mask, row], yp_arr[mask], alpha=0.75, s=50,
                           color=mc, edgecolors="white", lw=0.4, label=mat)
        lo = min(y_te[:,row].min(), yp_arr.min()) * 0.93
        hi = max(y_te[:,row].max(), yp_arr.max()) * 1.05
        ax.plot([lo,hi],[lo,hi],"k--",lw=1.3)
        r2   = r2_score(y_te[:,row], yp_arr)
        rmse = np.sqrt(mean_squared_error(y_te[:,row], yp_arr))
        ax.set_xlabel(f"Actual {LABELS[col]}", fontsize=9)
        ax.set_ylabel(f"Predicted {LABELS[col]}", fontsize=9)
        ax.set_title(f"{label}\n{col} | R2={r2:.4f}  RMSE={rmse:.4f}mm", fontsize=9)
        ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{PREFIX}_parity_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[OK] Saved: {PREFIX}_parity_plots.png")

# ── Fig 2: Residuals ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Residuals (Test Set) -- {TAG}", fontsize=13, fontweight="bold")
for ax, col, row in zip(axes, TARGETS, [0,1]):
    res = y_te[:,row] - yp_test[:,row]
    for mat, mc in MAT_COLORS.items():
        mask = mat_labels == mat
        if mask.sum():
            ax.scatter(yp_test[mask,row], res[mask], alpha=0.75, s=50,
                       color=mc, edgecolors="white", lw=0.4, label=mat)
    ax.axhline(0, color="black", linestyle="--", lw=1.3)
    ax.set_xlabel(f"Predicted {LABELS[col]}", fontsize=9)
    ax.set_ylabel("Residual mm", fontsize=9)
    ax.set_title(LABELS[col], fontsize=10); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{PREFIX}_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: {PREFIX}_residuals.png")

# ── Fig 3: Effect of Traverse Speed ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Effect of Traverse Speed -- {TAG} (SOD=10mm, Nozzle=1.57mm)",
             fontsize=13, fontweight="bold")
ts_range = np.linspace(480, 2160, N)
for mat, hv in MAT_HV.items():
    yp = hybrid_predict_raw(np.full(N,hv), np.full(N,1.57), ts_range, np.full(N,10.))
    for ax, col_idx in zip(axes, [0,1]):
        ax.plot(ts_range, yp[:,col_idx], lw=2.2, color=MAT_COLORS[mat], label=mat)
for ax, col in zip(axes, TARGETS):
    ax.set_xlabel("Traverse Speed (mm/min)", fontsize=10)
    ax.set_ylabel(LABELS[col], fontsize=10)
    ax.set_title(LABELS[col], fontsize=11); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{PREFIX}_effect_ts.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: {PREFIX}_effect_ts.png")

# ── Fig 4: Effect of SOD ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Effect of Stand-Off Distance -- {TAG} (TS=1200mm/min, Nozzle=1.57mm)",
             fontsize=13, fontweight="bold")
sod_range = np.linspace(3, 25, N)
for mat, hv in MAT_HV.items():
    yp = hybrid_predict_raw(np.full(N,hv), np.full(N,1.57), np.full(N,1200.), sod_range)
    for ax, col_idx in zip(axes, [0,1]):
        ax.plot(sod_range, yp[:,col_idx], lw=2.2, color=MAT_COLORS[mat], label=mat)
for ax, col in zip(axes, TARGETS):
    ax.set_xlabel("Stand-Off Distance (mm)", fontsize=10)
    ax.set_ylabel(LABELS[col], fontsize=10)
    ax.set_title(LABELS[col], fontsize=11); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{PREFIX}_effect_sod.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: {PREFIX}_effect_sod.png")

# ── Fig 5: Effect of Hardness ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Effect of Hardness -- {TAG} (TS=1200mm/min, SOD=10mm, Nozzle=1.57mm)",
             fontsize=13, fontweight="bold")
hv_range = np.linspace(350, 510, N)
yp = hybrid_predict_raw(hv_range, np.full(N,1.57), np.full(N,1200.), np.full(N,10.))
for ax, col_idx, col in zip(axes, [0,1], TARGETS):
    ax.plot(hv_range, yp[:,col_idx], lw=2.5, color=COLORS[col])
    for mat, hv in MAT_HV.items():
        ax.axvline(hv, color=MAT_COLORS[mat], linestyle="--", lw=1.2,
                   label=f"{mat}  Hv={hv}")
    ax.set_xlabel("Hardness (Hv)", fontsize=10)
    ax.set_ylabel(LABELS[col], fontsize=10)
    ax.set_title(LABELS[col], fontsize=11); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{PREFIX}_effect_hardness.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: {PREFIX}_effect_hardness.png")

# ── Fig 6: Effect of Nozzle Diameter ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Effect of Nozzle Diameter -- {TAG} (TS=1200mm/min, SOD=10mm)",
             fontsize=13, fontweight="bold")
d_range = np.linspace(0.5, 2.0, N)
for mat, hv in MAT_HV.items():
    yp = hybrid_predict_raw(np.full(N,hv), d_range, np.full(N,1200.), np.full(N,10.))
    for ax, col_idx in zip(axes, [0,1]):
        ax.plot(d_range, yp[:,col_idx], lw=2.2, color=MAT_COLORS[mat], label=mat)
for ax, col in zip(axes, TARGETS):
    ax.axvline(0.76, color="gray", linestyle=":", lw=1.2, label="0.76mm (data)")
    ax.axvline(1.57, color="gray", linestyle="--", lw=1.2, label="1.57mm (data)")
    ax.set_xlabel("Nozzle Diameter (mm)", fontsize=10)
    ax.set_ylabel(LABELS[col], fontsize=10)
    ax.set_title(LABELS[col], fontsize=11); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{PREFIX}_effect_nozzle.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: {PREFIX}_effect_nozzle.png")

# ── Fig 7: Physics decomposition ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Decomposition: Physics Baseline vs SVR Residual Contribution\n(Train Set)",
             fontsize=13, fontweight="bold")
for row, col in enumerate(TARGETS):
    axes[row][0].scatter(y_tr[:,row], phys_tr[:,row],
                         alpha=0.5, s=30, color="#6C63FF", edgecolors="white", lw=0.3)
    lo = y_tr[:,row].min()*0.9; hi = y_tr[:,row].max()*1.05
    axes[row][0].plot([lo,hi],[lo,hi],"k--",lw=1.3)
    r2_p = r2_score(y_tr[:,row], phys_tr[:,row])
    axes[row][0].set_title(f"Physics only | {LABELS[col]}\nR2={r2_p:.4f}", fontsize=9)
    axes[row][0].set_xlabel(f"Actual {LABELS[col]}"); axes[row][0].set_ylabel("Predicted")

    ml_contrib = yp_train[:,row] - phys_tr[:,row]
    axes[row][1].scatter(y_tr[:,row], ml_contrib,
                         alpha=0.5, s=30, color="#FF6584", edgecolors="white", lw=0.3)
    axes[row][1].axhline(0, color="black", lw=1.2, linestyle="--")
    axes[row][1].set_title(f"SVR Residual contribution | {LABELS[col]}", fontsize=9)
    axes[row][1].set_xlabel(f"Actual {LABELS[col]}")
    axes[row][1].set_ylabel("SVR Correction (mm)")
plt.tight_layout()
plt.savefig(f"{PREFIX}_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: {PREFIX}_decomposition.png")

# ── Fig 8: Feature importance ────────────────────────────────
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin

class HybridWrapper(BaseEstimator, RegressorMixin):
    def fit(self, X, y): return self
    def predict(self, X):
        df_tmp = pd.DataFrame(X, columns=FEATURES)
        return hybrid_predict_df(df_tmp)

pi = permutation_importance(HybridWrapper(), X_te, y_te,
                             n_repeats=30, random_state=42, scoring="r2")
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = ["#6C63FF","#FF6584","#43B89C","#FFA552"]
bars = ax.barh(FEATURES, pi.importances_mean, xerr=pi.importances_std,
               color=bar_colors, edgecolor="white", height=0.55,
               error_kw=dict(ecolor="gray", capsize=4))
ax.set_xlabel("Mean Decrease in R2 (permutation importance)", fontsize=10)
ax.set_title(f"Feature Importance -- {TAG}", fontsize=12, fontweight="bold")
ax.axvline(0, color="black", lw=0.8)
for bar, val in zip(bars, pi.importances_mean):
    ax.text(max(val,0)+0.002, bar.get_y()+bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{PREFIX}_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Saved: {PREFIX}_feature_importance.png")

# ======================================================================
# 10.  FINAL SUMMARY
# ======================================================================
print(f"\n{'='*62}")
print(f"FINAL HYBRID SVR MODEL -- TEST SET METRICS")
print(f"{'='*62}")
print(f"\n  Physics Baseline (pre-SVR correction):")
for i, col in enumerate(TARGETS):
    r2   = r2_score(y_te[:,i], phys_te[:,i])
    rmse = np.sqrt(mean_squared_error(y_te[:,i], phys_te[:,i]))
    print(f"    {col:16s}  R2={r2:.4f}  RMSE={rmse:.4f}mm")

print(f"\n  Physics + SVR Residual (final):")
for col in TARGETS:
    m = me[col]
    print(f"    {col:16s}  R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}mm  "
          f"MAE={m['MAE']:.4f}mm  MAPE={m['MAPE']:.2f}%")

print(f"\n  Best SVR params: {gs.best_params_}")
print(f"\n  Fitted power-law (DoC):")
print(f"    h = {p['K1']:.4e} * d^{p['alpha']:.3f} * v^{p['beta']:.3f} * SOD^{p['gamma']:.3f} * H^{p['delta']:.3f}")
print(f"  Fitted linear (KW):")
print(f"    W = {p['K2']:.4f}*d + {p['K3']:.4f}*SOD")
print(f"\n{'='*62}")
print(f"Saved files:")
print(f"  {PREFIX}_residual_model.pkl      <- SVR model")
print(f"  {PREFIX}_scaler_X.pkl            <- input scaler")
print(f"  {PREFIX}_scaler_r.pkl            <- residual scaler")
print(f"  {PREFIX}_physics_constants.json  <- K1..K3 + exponents")
print(f"{'='*62}")
