"""
AWJM Prediction Tool -- Streamlit Frontend
Supports: Physics-Hybrid ANN, SVR, GPR (Bayesian)
"""

import os, json, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import joblib
import tensorflow as tf

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AWJM Prediction Tool",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS (toned-down) ───────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  .stApp {
      background: #111827;
  }

  header[data-testid="stHeader"] { background: transparent !important; }
  .block-container { padding: 1.5rem 2.5rem 2.5rem 2.5rem; max-width: 1050px; margin: auto; }

  /* ── header area ── */
  .page-header {
      padding: 1.8rem 0 1.2rem 0;
      text-align: center;
  }
  .page-title {
      font-size: 1.8rem;
      font-weight: 700;
      color: #f3f4f6;
      margin-bottom: 0.3rem;
  }
  .page-sub {
      color: #9ca3af;
      font-size: 0.88rem;
      font-weight: 400;
  }

  /* ── section labels ── */
  .section-label {
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #6b7280;
      margin-bottom: 0.7rem;
  }

  /* ── card ── */
  .card {
      background: #1f2937;
      border: 1px solid #374151;
      border-radius: 12px;
      padding: 1.5rem 1.6rem;
  }

  /* ── widget overrides ── */
  div[data-testid="stNumberInput"] > label,
  div[data-testid="stSelectbox"] > label {
      color: #d1d5db !important;
      font-size: 0.82rem !important;
      font-weight: 500 !important;
  }
  div[data-testid="stNumberInput"] input {
      background: #111827 !important;
      border: 1px solid #374151 !important;
      border-radius: 8px !important;
      color: #f3f4f6 !important;
  }
  div[data-testid="stNumberInput"] input:focus {
      border-color: #6366f1 !important;
      box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
  }
  div[data-testid="stSelectbox"] > div > div {
      background: #111827 !important;
      border: 1px solid #374151 !important;
      border-radius: 8px !important;
      color: #f3f4f6 !important;
  }

  /* ── predict button ── */
  div[data-testid="stButton"] > button {
      width: 100%;
      padding: 0.7rem 1.5rem;
      background: #6366f1;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s ease;
  }
  div[data-testid="stButton"] > button:hover {
      background: #4f46e5;
  }

  /* ── result cards ── */
  .result-row {
      display: flex;
      gap: 1rem;
      margin-top: 0.6rem;
  }
  .result-card {
      flex: 1;
      border-radius: 10px;
      padding: 1.2rem 1rem;
      text-align: center;
  }
  .result-card-kw {
      background: #1e1b4b;
      border: 1px solid #312e81;
  }
  .result-card-doc {
      background: #1c1917;
      border: 1px solid #44403c;
  }
  .result-label {
      font-size: 0.68rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 0.4rem;
  }
  .result-label-kw { color: #a5b4fc; }
  .result-label-doc { color: #fbbf24; }
  .result-value {
      font-size: 2rem;
      font-weight: 700;
      line-height: 1;
      margin-bottom: 0.2rem;
  }
  .result-value-kw { color: #818cf8; }
  .result-value-doc { color: #f59e0b; }
  .result-unit {
      font-size: 0.75rem;
      color: #6b7280;
      font-weight: 500;
  }
  .result-std {
      font-size: 0.72rem;
      color: #9ca3af;
      margin-top: 0.25rem;
  }

  /* ── info chip ── */
  .chip {
      display: inline-block;
      background: #1f2937;
      border: 1px solid #374151;
      color: #9ca3af;
      font-size: 0.7rem;
      font-weight: 500;
      padding: 0.2rem 0.6rem;
      border-radius: 20px;
      margin: 0.12rem;
  }

  /* ── formula ── */
  .formula-box {
      background: #111827;
      border: 1px solid #374151;
      border-radius: 8px;
      padding: 0.8rem 1rem;
      font-family: 'Courier New', monospace;
      font-size: 0.75rem;
      color: #9ca3af;
      line-height: 1.9;
  }

  hr { border-color: #374151 !important; margin: 1rem 0 !important; }
  footer { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Load all models (cached) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_all_models():
    models = {}

    # ANN
    models["ann"] = dict(
        model=tf.keras.models.load_model("ann_residual_model.keras"),
        scaler_X=joblib.load("hybrid_scaler_X.pkl"),
        scaler_r=joblib.load("hybrid_scaler_r.pkl"),
        p=json.load(open("physics_constants.json")),
    )

    # SVR
    models["svr"] = dict(
        model=joblib.load("svr_residual_model.pkl"),
        scaler_X=joblib.load("svr_scaler_X.pkl"),
        scaler_r=joblib.load("svr_scaler_r.pkl"),
        p=json.load(open("svr_physics_constants.json")),
    )

    # GPR
    models["gpr"] = dict(
        model=joblib.load("gpr_residual_models.pkl"),   # dict of 2 GPRs
        scaler_X=joblib.load("gpr_scaler_X.pkl"),
        scaler_r=joblib.load("gpr_scaler_r.pkl"),
        p=json.load(open("gpr_physics_constants.json")),
    )

    return models

all_models = load_all_models()

# ── Physics baseline ──────────────────────────────────────────
def physics_predict(H, d, v, SOD, p):
    log_h = (np.log(p["K1"])
             + p["alpha"] * np.log(d)
             + p["beta"]  * np.log(v)
             + p["gamma"] * np.log(SOD)
             + p["delta"] * np.log(H))
    doc = np.exp(log_h)
    kw  = p["K2"] * d + p["K3"] * SOD
    return np.array([kw, doc])

# ── Prediction functions per model ────────────────────────────
def predict_ann(H, d, v, SOD):
    m = all_models["ann"]
    phys = physics_predict(H, d, v, SOD, m["p"])
    X = np.array([[H, d, v, SOD]], dtype=np.float32)
    X_s = m["scaler_X"].transform(X)
    resid_s = m["model"].predict(X_s, verbose=0)
    resid = m["scaler_r"].inverse_transform(resid_s)[0]
    result = phys + resid
    return float(result[0]), float(result[1]), None, None

def predict_svr(H, d, v, SOD):
    m = all_models["svr"]
    phys = physics_predict(H, d, v, SOD, m["p"])
    X = np.array([[H, d, v, SOD]], dtype=np.float32)
    X_s = m["scaler_X"].transform(X)
    resid_s = m["model"].predict(X_s)
    resid = m["scaler_r"].inverse_transform(resid_s)[0]
    result = phys + resid
    return float(result[0]), float(result[1]), None, None

def predict_gpr(H, d, v, SOD):
    m = all_models["gpr"]
    phys = physics_predict(H, d, v, SOD, m["p"])
    X = np.array([[H, d, v, SOD]], dtype=np.float32)
    X_s = m["scaler_X"].transform(X)
    gpr_kw = m["model"]["kerf_width"]
    gpr_doc = m["model"]["depth_of_cut"]
    resid_kw_s, std_kw_s = gpr_kw.predict(X_s, return_std=True)
    resid_doc_s, std_doc_s = gpr_doc.predict(X_s, return_std=True)
    resid_s = np.column_stack([resid_kw_s, resid_doc_s])
    resid = m["scaler_r"].inverse_transform(resid_s)[0]
    std = np.array([std_kw_s[0], std_doc_s[0]]) * m["scaler_r"].scale_
    result = phys + resid
    return float(result[0]), float(result[1]), float(std[0]), float(std[1])

MODEL_OPTIONS = {
    "Physics-Hybrid ANN": ("ann", predict_ann),
    "Physics-Hybrid SVR": ("svr", predict_svr),
    "Physics-Hybrid GPR (Bayesian)": ("gpr", predict_gpr),
}

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="page-header">
  <div class="page-title">AWJM Prediction Tool</div>
  <div class="page-sub">Predict kerf width and depth of cut using physics-hybrid machine learning</div>
</div>
""", unsafe_allow_html=True)

# Two columns
left_col, right_col = st.columns([1.05, 1], gap="large")

# ── LEFT: Inputs ──────────────────────────────────────────────
with left_col:
    #st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Process Parameters</div>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Prediction Model",
        list(MODEL_OPTIONS.keys()),
        help="ANN: neural network residual | SVR: support vector regression residual | GPR: Gaussian process with uncertainty"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        hardness = st.number_input(
            "Material Hardness (Hv)",
            min_value=100.0, max_value=1000.0,
            value=425.0, step=5.0, format="%.0f",
            help="Training range: 385-485 Hv"
        )
        traverse_speed = st.number_input(
            "Traverse Speed (mm/min)",
            min_value=1.0, max_value=10000.0,
            value=1200.0, step=60.0, format="%.0f",
            help="Training range: 480-3240 mm/min"
        )
    with c2:
        sod = st.number_input(
            "Stand-Off Distance (mm)",
            min_value=0.5, max_value=100.0,
            value=10.0, step=1.0, format="%.1f",
            help="Training range: 3-25 mm"
        )
        nozzle_dia = st.number_input(
            "Nozzle Diameter (mm)",
            min_value=0.1, max_value=5.0,
            value=1.57, step=0.01, format="%.2f",
            help="Training data: 0.76 mm and 1.57 mm"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT: Results + Model info ───────────────────────────────
with right_col:
    #st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)

    if predict_clicked:
        with st.spinner("Running model..."):
            try:
                _, predict_fn = MODEL_OPTIONS[model_choice]
                kw, doc, std_kw, std_doc = predict_fn(
                    hardness, nozzle_dia, traverse_speed, sod
                )
                kw = max(kw, 0.0)
                doc = max(doc, 0.0)

                # Uncertainty line (GPR only)
                std_html_kw = ""
                std_html_doc = ""
                if std_kw is not None:
                    std_html_kw = f'<div class="result-std">&plusmn; {std_kw:.3f} mm (1&sigma;)</div>'
                    std_html_doc = f'<div class="result-std">&plusmn; {std_doc:.3f} mm (1&sigma;)</div>'

                st.markdown(f"""
                <div class="result-row">
                  <div class="result-card result-card-kw">
                    <div class="result-label result-label-kw">Kerf Width</div>
                    <div class="result-value result-value-kw">{kw:.3f}</div>
                    <div class="result-unit">mm</div>
                    {std_html_kw}
                  </div>
                  <div class="result-card result-card-doc">
                    <div class="result-label result-label-doc">Depth of Cut</div>
                    <div class="result-value result-value-doc">{doc:.3f}</div>
                    <div class="result-unit">mm</div>
                    {std_html_doc}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Breakdown expander
                model_key, _ = MODEL_OPTIONS[model_choice]
                phys = physics_predict(hardness, nozzle_dia, traverse_speed, sod,
                                       all_models[model_key]["p"])
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Physics baseline vs ML correction"):
                    b1, b2 = st.columns(2)
                    with b1:
                        st.markdown("**Kerf Width**")
                        st.metric("Physics baseline", f"{phys[0]:.3f} mm")
                        st.metric("ML correction", f"{kw - phys[0]:+.3f} mm")
                        st.metric("Final", f"{kw:.3f} mm")
                    with b2:
                        st.markdown("**Depth of Cut**")
                        st.metric("Physics baseline", f"{phys[1]:.3f} mm")
                        st.metric("ML correction", f"{doc - phys[1]:+.3f} mm")
                        st.metric("Final", f"{doc:.3f} mm")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 2.5rem 1rem; color: #6b7280;">
          <div style="font-size: 0.9rem;">
            Set parameters and press <b>Predict</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    model_key, _ = MODEL_OPTIONS[model_choice]
    p = all_models[model_key]["p"]



    st.markdown('</div>', unsafe_allow_html=True)
