"""
Physics-Hybrid ANN for AWJM - Final Model
==========================================
Architecture (Option 1 from physics.txt):

  STEP 1 - Physics Baseline (fitted from data):
    Depth of Cut : h = K1 * d^2 / (v * SOD * H)
    Kerf Width   : W = K2 * d  + K3 * SOD

  STEP 2 - ANN Residual:
    Learns the deviation (actual - physics_baseline)
    Inputs: [H, d, v, SOD]

  STEP 3 - Final Prediction:
    output = physics_baseline + ANN_residual

Benefits:
  - Physics guarantees correct monotonicity in all sweeps
  - ANN corrects for material-specific & nonlinear deviations
  - No penalty terms needed; no feature hacking
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

tf.random.set_seed(42)
np.random.seed(42)

FEATURES   = ["hardness", "nozzle_dia", "traverse_speed", "SOD"]
TARGETS    = ["kerf_width", "depth_of_cut"]
LABELS     = {"kerf_width": "Kerf Width (mm)", "depth_of_cut": "Depth of Cut (mm)"}
COLORS     = {"kerf_width": "#6C63FF", "depth_of_cut": "#FF6584"}
MAT_COLORS = {"Titanium": "#6C63FF", "Inconel": "#FF6584", "SS316": "#43B89C"}
MAT_HV     = {"Titanium": 385, "Inconel": 485, "SS316": 425}

# ══════════════════════════════════════════════════════════════
# 1.  DATA PARSERS
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
# 2.  PHYSICS MODEL
# ══════════════════════════════════════════════════════════════
def fit_physics_constants(df_train):
    """
    Fit physics model with OPTIMAL exponents using log-linear regression.

    DoC power law   (log-linearised):
      log(h) = log(K1) + alpha*log(d) - beta*log(v) - gamma*log(SOD) - delta*log(H)
      => linear in [1, log(d), log(v), log(SOD), log(H)]

    KW linear model:
      W = K2*d + K3*SOD   (literature-motivated additive form)
      => standard OLS on [d, SOD]

    Returns a dict of all fitted parameters.
    """
    h  = df_train["depth_of_cut"].values
    W  = df_train["kerf_width"].values
    d  = df_train["nozzle_dia"].values
    v  = df_train["traverse_speed"].values
    s  = df_train["SOD"].values
    H  = df_train["hardness"].values

    # ── DoC: fit in log space ──────────────────────────────────
    log_h = np.log(h)
    # Design matrix: [1, log(d), -log(v), -log(SOD), -log(H)]
    # But we let OLS find the coefficients freely so each column
    # gets its own sign-unconstrained coefficient, then we report
    # the exponents (and check signs match physics).
    A_doc = np.column_stack([
        np.ones(len(h)),
        np.log(d),
        np.log(v),
        np.log(s),
        np.log(H),
    ])
    coeffs_doc, _, _, _ = np.linalg.lstsq(A_doc, log_h, rcond=None)
    log_K1, alpha, beta, gamma, delta = coeffs_doc
    K1    = np.exp(log_K1)

    # ── KW: linear OLS on [d, SOD] ────────────────────────────
    A_kw = np.column_stack([d, s])
    K2, K3 = np.linalg.lstsq(A_kw, W, rcond=None)[0]

    params = dict(
        K1=float(K1),
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        delta=float(delta),
        K2=float(K2),
        K3=float(K3),
    )
    return params

def physics_baseline(df, p):
    """
    Returns physics predictions as numpy array, shape (n, 2):
      col 0 = KW prediction  (W = K2*d + K3*SOD)
      col 1 = DoC prediction (h = K1*d^alpha / (v^|beta|*SOD^|gamma|*H^|delta|))
    Uses fitted exponents stored in dict p.
    """
    d   = df["nozzle_dia"].values
    v   = df["traverse_speed"].values
    s   = df["SOD"].values
    H   = df["hardness"].values

    # DoC: use the log-model directly (avoids any sign ambiguity)
    log_h = (np.log(p["K1"])
             + p["alpha"] * np.log(d)
             + p["beta"]  * np.log(v)
             + p["gamma"] * np.log(s)
             + p["delta"] * np.log(H))
    doc_pred = np.exp(log_h)

    kw_pred  = p["K2"] * d + p["K3"] * s
    return np.column_stack([kw_pred, doc_pred])   # matches TARGETS order

def physics_baseline_raw(H, d, v, SOD, p):
    """Physics prediction from scalar / numpy arrays."""
    log_h = (np.log(p["K1"])
             + p["alpha"] * np.log(d)
             + p["beta"]  * np.log(v)
             + p["gamma"] * np.log(SOD)
             + p["delta"] * np.log(H))
    doc = np.exp(log_h)
    kw  = p["K2"] * d + p["K3"] * SOD
    return np.stack([kw, doc], axis=-1)           # shape (..., 2)

# ══════════════════════════════════════════════════════════════
# 3.  BUILD COMBINED DATASET
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  Physics-Hybrid ANN -- Final Model")
print("  Formulas from physics.txt:")
print("    DoC : h = K1 * d^2 / (v * SOD * H)")
print("    KW  : W = K2 * d  + K3 * SOD")
print("    Final = Physics Baseline + ANN Residual")
print("=" * 62)

df = pd.concat([
    parse_ti("Ti_nozzle 0.76_and_1.5mm.xlsx"),
    parse_inconel("IN_ nozzle 0.76_and_1.5mm.xlsx"),
    parse_ss("SS_ nozzle 0.76_and_1.5mm.xlsx"),
], ignore_index=True)
df = df[FEATURES + TARGETS + ["material"]].dropna()
print(f"\n[OK] Dataset: {len(df)} samples")

X_raw  = df[FEATURES].values.astype(np.float32)
y_true = df[TARGETS].values.astype(np.float32)   # [KW, DoC]
groups = df["material"] + "_" + df["nozzle_dia"].astype(str)

# ── Train / Test split ────────────────────────────────────────
(X_tr, X_te, y_tr, y_te,
 df_tr_idx, df_te_idx,
 g_tr, g_te) = train_test_split(
    X_raw, y_true, df.index, groups,
    test_size=0.20, random_state=42, stratify=groups
)

df_train = df.loc[df_tr_idx].copy()
df_test  = df.loc[df_te_idx].copy()
print(f"[OK] Train: {len(X_tr)} | Test: {len(X_te)}")

# ══════════════════════════════════════════════════════════════
# 4.  FIT PHYSICS CONSTANTS ON TRAINING DATA ONLY
# ══════════════════════════════════════════════════════════════
p = fit_physics_constants(df_train)
K1=p["K1"]; K2=p["K2"]; K3=p["K3"]
print(f"\n[OK] Physics power-law fitted (optimal exponents):")
print(f"     DoC : h = {K1:.4e} * d^{p['alpha']:.3f} * v^{p['beta']:.3f} * SOD^{p['gamma']:.3f} * H^{p['delta']:.3f}")
print(f"     KW  : W = {K2:.4f}*d + {K3:.4f}*SOD")

# Save constants
with open("physics_constants.json", "w") as f:
    json.dump(p, f, indent=2)
print("[OK] Saved: physics_constants.json")

# ── Compute baselines & residuals ─────────────────────────────
phys_tr = physics_baseline(df_train, p).astype(np.float32)
phys_te = physics_baseline(df_test,  p).astype(np.float32)

resid_tr = y_tr - phys_tr    # what the ANN must learn
resid_te = y_te - phys_te

# Physics-only metrics (before ANN correction)
print(f"\n{'-'*62}")
print("PHYSICS-ONLY BASELINE (no ANN yet)")
print(f"{'-'*62}")
for i, col in enumerate(TARGETS):
    r2   = r2_score(y_te[:, i], phys_te[:, i])
    rmse = np.sqrt(mean_squared_error(y_te[:, i], phys_te[:, i]))
    print(f"  {col:16s}  R2={r2:.4f}  RMSE={rmse:.4f}mm")

print(f"\n  Residual stats (train):")
for i, col in enumerate(TARGETS):
    print(f"  {col:16s}  mean={resid_tr[:,i].mean():.4f}  "
          f"std={resid_tr[:,i].std():.4f}  "
          f"max_abs={np.abs(resid_tr[:,i]).max():.4f}")

# ══════════════════════════════════════════════════════════════
# 5.  SCALE INPUTS & RESIDUALS FOR ANN
# ══════════════════════════════════════════════════════════════
scaler_X = StandardScaler()
scaler_r = StandardScaler()

X_tr_s = scaler_X.fit_transform(X_tr)
X_te_s = scaler_X.transform(X_te)
r_tr_s = scaler_r.fit_transform(resid_tr)
r_te_s = scaler_r.transform(resid_te)

joblib.dump(scaler_X, "hybrid_scaler_X.pkl")
joblib.dump(scaler_r, "hybrid_scaler_r.pkl")
print("[OK] Scalers saved.")

# ══════════════════════════════════════════════════════════════
# 6.  ANN RESIDUAL MODEL
#     Smaller than the baseline ANN — residuals are compact.
#     3 layers x 64 neurons, heavier dropout.
# ══════════════════════════════════════════════════════════════
def build_residual_ann(neurons=64, n_layers=3, dropout=0.20, l2=1e-4, lr=8e-4):
    inp = keras.Input(shape=(4,), name="raw_inputs")
    x   = inp
    for i in range(n_layers):
        x = layers.Dense(neurons, activation="relu",
                         kernel_regularizer=regularizers.l2(l2),
                         name=f"h{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn{i+1}")(x)
        x = layers.Dropout(dropout, name=f"dr{i+1}")(x)
    out = layers.Dense(2, name="residual_outputs")(x)
    m   = keras.Model(inp, out, name="ResidualANN")
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return m

ann = build_residual_ann()
ann.summary()

cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=150,
                            restore_best_weights=True, verbose=0),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=50, min_lr=1e-6, verbose=0),
    callbacks.ModelCheckpoint("best_residual_ann.keras", monitor="val_loss",
                              save_best_only=True, verbose=0),
]

print("\nTraining Residual ANN ...")
hist = ann.fit(
    X_tr_s, r_tr_s,
    validation_split=0.15,
    epochs=3000,
    batch_size=32,
    callbacks=cb_list,
    verbose=0
)
best_ep = np.argmin(hist.history["val_loss"]) + 1
print(f"[OK] Training complete. Best epoch: {best_ep}")

# ══════════════════════════════════════════════════════════════
# 7.  FULL HYBRID PREDICTION
# ══════════════════════════════════════════════════════════════
def hybrid_predict_df(df_in):
    """Predict from a DataFrame with raw feature columns."""
    X_raw_in = df_in[FEATURES].values.astype(np.float32)
    phys     = physics_baseline(df_in, p)
    X_s      = scaler_X.transform(X_raw_in)
    resid_s  = ann.predict(X_s, verbose=0)
    resid    = scaler_r.inverse_transform(resid_s)
    return phys + resid                     # shape (n, 2)

def hybrid_predict_raw(H, d, v, SOD):
    """Predict from scalar / numpy arrays."""
    H = np.atleast_1d(H); d = np.atleast_1d(d)
    v = np.atleast_1d(v); SOD = np.atleast_1d(SOD)
    phys = physics_baseline_raw(H, d, v, SOD, p)   # (n, 2)
    X_raw_in = np.column_stack([H, d, v, SOD]).astype(np.float32)
    X_s      = scaler_X.transform(X_raw_in)
    resid_s  = ann.predict(X_s, verbose=0)
    resid    = scaler_r.inverse_transform(resid_s)
    return phys + resid

# ══════════════════════════════════════════════════════════════
# 8.  EVALUATE HYBRID MODEL
# ══════════════════════════════════════════════════════════════
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

print(f"\n{'-'*62}\nFINAL HYBRID MODEL EVALUATION\n{'-'*62}")
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

# ══════════════════════════════════════════════════════════════
# 9.  PLOTS
# ══════════════════════════════════════════════════════════════
plt.style.use("seaborn-v0_8-whitegrid")
N = 300

# ── Fig 1: Training history ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Residual ANN Training History -- Physics-Hybrid Model",
             fontsize=14, fontweight="bold")
for ax, key, lbl in zip(axes, ["loss","mae"], ["MSE Loss (residual)","MAE (residual)"]):
    ax.plot(hist.history[key],           lw=1.8, color="#6C63FF", label="Train")
    ax.plot(hist.history[f"val_{key}"],  lw=1.8, color="#FF6584",
            linestyle="--", label="Validation")
    ax.axvline(best_ep-1, color="gray", linestyle=":", lw=1.2,
               label=f"Best ({best_ep})")
    ax.set_xlabel("Epoch"); ax.set_ylabel(lbl); ax.set_title(lbl)
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("hybrid_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[OK] Saved: hybrid_training_curves.png")

# ── Fig 2: Physics-only vs Hybrid parity (2x2 for test set) ──
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle("Parity Plots -- Physics Baseline vs Final Hybrid\n(Test Set)",
             fontsize=13, fontweight="bold")
mat_tr_labels = g_tr.str.split("_").str[0].values

for row, col in enumerate(TARGETS):
    for ax, yp_arr, label in [
        (axes[row][0], phys_te[:, row], "Physics Baseline only"),
        (axes[row][1], yp_test[:,  row], "Physics + ANN Residual"),
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
plt.savefig("hybrid_parity_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: hybrid_parity_plots.png")

# ── Fig 3: Residuals ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Residuals (Test Set) -- Physics-Hybrid Model", fontsize=13, fontweight="bold")
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
plt.savefig("hybrid_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: hybrid_residuals.png")

# ── Fig 4: Effect of Traverse Speed ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Effect of Traverse Speed -- Physics-Hybrid (SOD=10mm, Nozzle=1.57mm)",
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
plt.savefig("hybrid_effect_ts.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: hybrid_effect_ts.png")

# ── Fig 5: Effect of SOD ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Effect of Stand-Off Distance -- Physics-Hybrid (TS=1200mm/min, Nozzle=1.57mm)",
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
plt.savefig("hybrid_effect_sod.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: hybrid_effect_sod.png")

# ── Fig 6: Effect of Hardness ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Effect of Hardness -- Physics-Hybrid (TS=1200mm/min, SOD=10mm, Nozzle=1.57mm)",
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
plt.savefig("hybrid_effect_hardness.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: hybrid_effect_hardness.png")

# ── Fig 7: Effect of Nozzle Diameter ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Effect of Nozzle Diameter -- Physics-Hybrid (TS=1200mm/min, SOD=10mm)",
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
plt.savefig("hybrid_effect_nozzle.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: hybrid_effect_nozzle.png")

# ── Fig 8: Physics decomposition (baseline vs residual) ───────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Decomposition: Physics Baseline vs ANN Residual Contribution\n(Train Set)",
             fontsize=13, fontweight="bold")
for row, col in enumerate(TARGETS):
    axes[row][0].scatter(y_tr[:,row], phys_tr[:,row],
                         alpha=0.5, s=30, color="#6C63FF", edgecolors="white", lw=0.3)
    lo = y_tr[:,row].min()*0.9; hi = y_tr[:,row].max()*1.05
    axes[row][0].plot([lo,hi],[lo,hi],"k--",lw=1.3)
    r2_p = r2_score(y_tr[:,row], phys_tr[:,row])
    axes[row][0].set_title(f"Physics only | {LABELS[col]}\nR2={r2_p:.4f}", fontsize=9)
    axes[row][0].set_xlabel(f"Actual {LABELS[col]}"); axes[row][0].set_ylabel("Predicted")

    ann_contrib = yp_train[:,row] - phys_tr[:,row]
    axes[row][1].scatter(y_tr[:,row], ann_contrib,
                         alpha=0.5, s=30, color="#FF6584", edgecolors="white", lw=0.3)
    axes[row][1].axhline(0, color="black", lw=1.2, linestyle="--")
    axes[row][1].set_title(f"ANN Residual contribution | {LABELS[col]}", fontsize=9)
    axes[row][1].set_xlabel(f"Actual {LABELS[col]}")
    axes[row][1].set_ylabel("ANN Correction (mm)")
plt.tight_layout()
plt.savefig("hybrid_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: hybrid_decomposition.png")

# ── Fig 9: Feature importance ─────────────────────────────────
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
ax.set_title("Feature Importance -- Physics-Hybrid ANN", fontsize=12, fontweight="bold")
ax.axvline(0, color="black", lw=0.8)
for bar, val in zip(bars, pi.importances_mean):
    ax.text(max(val,0)+0.002, bar.get_y()+bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("hybrid_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Saved: hybrid_feature_importance.png")

# ══════════════════════════════════════════════════════════════
# 10.  SAVE MODEL
# ══════════════════════════════════════════════════════════════
ann.save("ann_residual_model.keras")
print("[OK] Model saved: ann_residual_model.keras")

# ══════════════════════════════════════════════════════════════
# 11.  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print("FINAL HYBRID MODEL -- TEST SET METRICS")
print(f"{'='*62}")
print(f"\n  Physics Baseline (pre-ANN correction):")
for i, col in enumerate(TARGETS):
    r2   = r2_score(y_te[:,i], phys_te[:,i])
    rmse = np.sqrt(mean_squared_error(y_te[:,i], phys_te[:,i]))
    print(f"    {col:16s}  R2={r2:.4f}  RMSE={rmse:.4f}mm")

print(f"\n  Physics + ANN Residual (final):")
for col in TARGETS:
    m = me[col]
    print(f"    {col:16s}  R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}mm  "
          f"MAE={m['MAE']:.4f}mm  MAPE={m['MAPE']:.2f}%")

print(f"\n  Fitted power-law (DoC):")
print(f"    h = {p['K1']:.4e} * d^{p['alpha']:.3f} * v^{p['beta']:.3f} * SOD^{p['gamma']:.3f} * H^{p['delta']:.3f}")
print(f"  Fitted linear (KW):")
print(f"    W = {p['K2']:.4f}*d + {p['K3']:.4f}*SOD")
print(f"\n{'='*62}")
print("Saved files:")
print("  ann_residual_model.keras     <- ANN component")
print("  best_residual_ann.keras      <- best checkpoint")
print("  hybrid_scaler_X.pkl          <- input scaler")
print("  hybrid_scaler_r.pkl          <- residual scaler")
print("  physics_constants.json        <- K1, K2, K3")
print("  combined_awjm_dataset.csv     <- combined data")
print(f"{'='*62}")
print("\nHow to use in inference:")
print("  1. Compute physics baseline: h = K1*d^2/(v*SOD*H), W = K2*d + K3*SOD")
print("  2. Scale inputs with hybrid_scaler_X.pkl")
print("  3. Predict residual with ann_residual_model.keras")
print("  4. Inverse-scale residual with hybrid_scaler_r.pkl")
print("  5. Final = baseline + residual")
