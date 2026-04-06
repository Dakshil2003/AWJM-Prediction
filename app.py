"""
AWJM Prediction Tool — Streamlit Frontend
Physics-Hybrid ANN: Physics Baseline + ANN Residual
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

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* ---------- global background ---------- */
  .stApp {
      background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
      min-height: 100vh;
  }

  /* ---------- hide default header ---------- */
  header[data-testid="stHeader"] { background: transparent !important; }
  .block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1100px; margin: auto; }

  /* ---------- hero ---------- */
  .hero-wrap {
      text-align: center;
      padding: 2.8rem 1rem 1.8rem 1rem;
  }
  .hero-badge {
      display: inline-block;
      background: linear-gradient(90deg, #6C63FF22, #FF658422);
      border: 1px solid #6C63FF55;
      color: #a89fff;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      padding: 0.35rem 1.1rem;
      border-radius: 50px;
      margin-bottom: 1.1rem;
  }
  .hero-title {
      font-size: 2.8rem;
      font-weight: 800;
      background: linear-gradient(135deg, #ffffff 30%, #a89fff 70%, #FF6584 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.15;
      margin-bottom: 0.7rem;
  }
  .hero-sub {
      color: #8892b0;
      font-size: 1.0rem;
      font-weight: 400;
      max-width: 620px;
      margin: 0 auto 0.5rem auto;
      line-height: 1.65;
  }

  /* ---------- section labels ---------- */
  .section-label {
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #6C63FF;
      margin-bottom: 0.9rem;
      padding-left: 0.1rem;
  }

  /* ---------- glass card ---------- */
  .glass-card {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.09);
      border-radius: 18px;
      padding: 1.8rem 2rem;
      backdrop-filter: blur(12px);
  }

  /* ---------- Streamlit widget overrides ---------- */
  div[data-testid="stNumberInput"] > label,
  div[data-testid="stSelectbox"] > label {
      color: #ccd6f6 !important;
      font-size: 0.83rem !important;
      font-weight: 500 !important;
  }
  div[data-testid="stNumberInput"] input {
      background: rgba(255,255,255,0.06) !important;
      border: 1px solid rgba(108,99,255,0.35) !important;
      border-radius: 10px !important;
      color: #e6f1ff !important;
      font-size: 1rem !important;
      font-weight: 500 !important;
  }
  div[data-testid="stNumberInput"] input:focus {
      border-color: #6C63FF !important;
      box-shadow: 0 0 0 3px rgba(108,99,255,0.20) !important;
  }
  div[data-testid="stSelectbox"] > div > div {
      background: rgba(255,255,255,0.06) !important;
      border: 1px solid rgba(108,99,255,0.35) !important;
      border-radius: 10px !important;
      color: #e6f1ff !important;
  }

  /* ---------- predict button ---------- */
  div[data-testid="stButton"] > button {
      width: 100%;
      padding: 0.85rem 2rem;
      background: linear-gradient(135deg, #6C63FF, #8B5CF6);
      color: white;
      border: none;
      border-radius: 12px;
      font-size: 1.05rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      cursor: pointer;
      transition: all 0.25s ease;
      box-shadow: 0 4px 20px rgba(108,99,255,0.40);
      margin-top: 0.5rem;
  }
  div[data-testid="stButton"] > button:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(108,99,255,0.55);
      background: linear-gradient(135deg, #7c74ff, #9B6CF6);
  }
  div[data-testid="stButton"] > button:active {
      transform: translateY(0px);
  }

  /* ---------- result cards ---------- */
  .result-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.2rem;
      margin-top: 0.5rem;
  }
  .result-card {
      border-radius: 16px;
      padding: 1.8rem 1.6rem;
      text-align: center;
      position: relative;
      overflow: hidden;
  }
  .result-card-kw {
      background: linear-gradient(135deg, #1a1640 0%, #2d2060 100%);
      border: 1px solid rgba(108,99,255,0.45);
      box-shadow: 0 4px 30px rgba(108,99,255,0.20);
  }
  .result-card-doc {
      background: linear-gradient(135deg, #1f1025 0%, #3a1535 100%);
      border: 1px solid rgba(255,101,132,0.45);
      box-shadow: 0 4px 30px rgba(255,101,132,0.20);
  }
  .result-icon { font-size: 2rem; margin-bottom: 0.5rem; }
  .result-label {
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      margin-bottom: 0.6rem;
  }
  .result-label-kw { color: #a89fff; }
  .result-label-doc { color: #ff9eb5; }
  .result-value {
      font-size: 2.6rem;
      font-weight: 800;
      line-height: 1;
      margin-bottom: 0.3rem;
  }
  .result-value-kw {
      background: linear-gradient(135deg, #a89fff, #6C63FF);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
  }
  .result-value-doc {
      background: linear-gradient(135deg, #ff9eb5, #FF6584);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
  }
  .result-unit {
      font-size: 0.82rem;
      color: #5a6480;
      font-weight: 500;
  }

  /* ---------- formula box ---------- */
  .formula-box {
      background: rgba(108,99,255,0.07);
      border: 1px solid rgba(108,99,255,0.22);
      border-radius: 12px;
      padding: 1.0rem 1.3rem;
      font-family: 'Courier New', monospace;
      font-size: 0.80rem;
      color: #a89fff;
      line-height: 2.0;
      margin-top: 0.5rem;
  }

  /* ---------- info chip ---------- */
  .chip {
      display: inline-block;
      background: rgba(108,99,255,0.15);
      border: 1px solid rgba(108,99,255,0.30);
      color: #a89fff;
      font-size: 0.73rem;
      font-weight: 600;
      padding: 0.22rem 0.75rem;
      border-radius: 50px;
      margin: 0.15rem;
  }

  /* ---------- warning / range ---------- */
  .range-warn {
      background: rgba(255,165,82,0.10);
      border: 1px solid rgba(255,165,82,0.35);
      border-radius: 10px;
      color: #ffa552;
      font-size: 0.78rem;
      padding: 0.6rem 0.9rem;
      margin-top: 0.7rem;
  }

  /* ---------- divider ---------- */
  hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

  /* hide streamlit footer */
  footer { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    ann      = tf.keras.models.load_model("ann_residual_model.keras")
    scaler_X = joblib.load("hybrid_scaler_X.pkl")
    scaler_r = joblib.load("hybrid_scaler_r.pkl")
    with open("physics_constants.json") as f:
        p = json.load(f)
    return ann, scaler_X, scaler_r, p

ann, scaler_X, scaler_r, p = load_model()

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

def full_predict(H, d, v, SOD):
    phys     = physics_predict(H, d, v, SOD, p)
    X_raw    = np.array([[H, d, v, SOD]], dtype=np.float32)
    X_s      = scaler_X.transform(X_raw)
    resid_s  = ann.predict(X_s, verbose=0)
    resid    = scaler_r.inverse_transform(resid_s)[0]
    result   = phys + resid
    return float(result[0]), float(result[1])   # kw, doc

# ── Training ranges for validation ───────────────────────────
RANGES = {
    "hardness":       (385, 485),
    "nozzle_dia":     (0.76, 1.57),
    "traverse_speed": (480, 3240),
    "sod":            (3, 25),
}

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">Physics-Hybrid AI &nbsp;•&nbsp; AWJM</div>
  <div class="hero-title">Abrasive Water Jet<br>Machining Predictor</div>
  <div class="hero-sub">
    Predict Kerf Width &amp; Depth of Cut instantly from your process parameters
    using a physics-informed hybrid neural network.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────
left_col, right_col = st.columns([1.05, 1], gap="large")

# ── LEFT: Inputs ──────────────────────────────────────────────
with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Process Parameters</div>', unsafe_allow_html=True)

    # Model selector
    model_choice = st.selectbox(
        "Prediction Model",
        ["Physics-Hybrid ANN  (DoC power-law + KW linear + Residual ANN)"],
        help="More models can be added in future versions."
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        hardness = st.number_input(
            "Material Hardness (Hv)",
            min_value=100.0, max_value=1000.0,
            value=425.0, step=5.0,
            format="%.0f",
            help="Vickers hardness of the workpiece material. "
                 "Training range: 385–485 Hv"
        )
        traverse_speed = st.number_input(
            "Traverse Speed (mm/min)",
            min_value=1.0, max_value=10000.0,
            value=1200.0, step=60.0,
            format="%.0f",
            help="Speed at which the nozzle moves across the workpiece. "
                 "Training range: 480–3240 mm/min"
        )

    with c2:
        sod = st.number_input(
            "Stand-Off Distance (mm)",
            min_value=0.5, max_value=100.0,
            value=10.0, step=1.0,
            format="%.1f",
            help="Distance between nozzle tip and workpiece surface. "
                 "Training range: 3–25 mm"
        )
        nozzle_dia = st.number_input(
            "Nozzle Diameter (mm)",
            min_value=0.1, max_value=5.0,
            value=1.57, step=0.01,
            format="%.2f",
            help="Inner diameter of the waterjet nozzle. "
                 "Training data: 0.76 mm and 1.57 mm"
        )

    # Out-of-range warning
    out_of_range = []
    if not (RANGES["hardness"][0] <= hardness <= RANGES["hardness"][1]):
        out_of_range.append(f"Hardness {hardness:.0f} Hv  (trained: 385–485)")
    if not (RANGES["nozzle_dia"][0] <= nozzle_dia <= RANGES["nozzle_dia"][1]):
        out_of_range.append(f"Nozzle dia {nozzle_dia:.2f} mm  (trained: 0.76–1.57)")
    if not (RANGES["traverse_speed"][0] <= traverse_speed <= RANGES["traverse_speed"][1]):
        out_of_range.append(f"Traverse speed {traverse_speed:.0f} mm/min  (trained: 480–3240)")
    if not (RANGES["sod"][0] <= sod <= RANGES["sod"][1]):
        out_of_range.append(f"SOD {sod:.1f} mm  (trained: 3–25)")

    if out_of_range:
        warn_html = "<div class='range-warn'>⚠️ <b>Extrapolation warning</b> — outside training range:<br>" + \
                    "<br>".join(f"&nbsp;&nbsp;• {w}" for w in out_of_range) + \
                    "<br><small>Physics baseline still applies, but ANN residual may be unreliable.</small></div>"
        st.markdown(warn_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button("⚡  Predict", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT: Results + Model info ───────────────────────────────
with right_col:

    # Results panel
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)

    if predict_clicked:
        with st.spinner("Running model..."):
            try:
                kw, doc = full_predict(hardness, nozzle_dia, traverse_speed, sod)

                # Clamp to physically realistic bounds
                kw  = max(kw,  0.0)
                doc = max(doc, 0.0)

                st.markdown(f"""
                <div class="result-grid">
                  <div class="result-card result-card-kw">
                    <div class="result-icon">📐</div>
                    <div class="result-label result-label-kw">Kerf Width</div>
                    <div class="result-value result-value-kw">{kw:.3f}</div>
                    <div class="result-unit">millimetres</div>
                  </div>
                  <div class="result-card result-card-doc">
                    <div class="result-icon">⬇️</div>
                    <div class="result-label result-label-doc">Depth of Cut</div>
                    <div class="result-value result-value-doc">{doc:.3f}</div>
                    <div class="result-unit">millimetres</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Physics baseline breakdown
                phys = physics_predict(hardness, nozzle_dia, traverse_speed, sod, p)
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Show physics baseline vs ANN correction"):
                    b1, b2 = st.columns(2)
                    with b1:
                        st.markdown("**Kerf Width**")
                        st.metric("Physics baseline", f"{phys[0]:.3f} mm")
                        st.metric("ANN correction",   f"{kw - phys[0]:+.3f} mm")
                        st.metric("Final",            f"{kw:.3f} mm")
                    with b2:
                        st.markdown("**Depth of Cut**")
                        st.metric("Physics baseline", f"{phys[1]:.3f} mm")
                        st.metric("ANN correction",   f"{doc - phys[1]:+.3f} mm")
                        st.metric("Final",            f"{doc:.3f} mm")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #4a5568;">
          <div style="font-size:3rem; margin-bottom:1rem;">💧</div>
          <div style="font-size:1.0rem; font-weight:500; color:#5a6480;">
            Set your process parameters<br>and press <b style="color:#6C63FF;">Predict</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model info card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Model Architecture</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="formula-box">
      DoC &nbsp;= &nbsp;<b>K₁ × d<sup>α</sup> × v<sup>β</sup> × SOD<sup>γ</sup> × H<sup>δ</sup></b> &nbsp;+ &nbsp;ANN residual<br>
      KW &nbsp;&nbsp;= &nbsp;<b>K₂ × d &nbsp;+ &nbsp;K₃ × SOD</b> &nbsp;+ &nbsp;ANN residual<br><br>
      K₁&nbsp;=&nbsp;{p["K1"]:.3e} &nbsp;|&nbsp;
      α&nbsp;=&nbsp;{p["alpha"]:.3f} &nbsp;|&nbsp;
      β&nbsp;=&nbsp;{p["beta"]:.3f}<br>
      γ&nbsp;=&nbsp;{p["gamma"]:.3f} &nbsp;|&nbsp;
      δ&nbsp;=&nbsp;{p["delta"]:.3f}<br>
      K₂&nbsp;=&nbsp;{p["K2"]:.4f} &nbsp;|&nbsp;
      K₃&nbsp;=&nbsp;{p["K3"]:.4f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="color:#5a6480; font-size:0.78rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;">Training Data</div>', unsafe_allow_html=True)
    st.markdown("""
    <span class="chip">Titanium &nbsp;Hv=385</span>
    <span class="chip">SS316 &nbsp;Hv=425</span>
    <span class="chip">Inconel &nbsp;Hv=485</span>
    <br>
    <span class="chip">Nozzle 0.76 mm</span>
    <span class="chip">Nozzle 1.57 mm</span>
    <span class="chip">541 samples</span>
    <br>
    <span class="chip">Pressure 35 ksi</span>
    <span class="chip">DoC R²&nbsp;=&nbsp;0.986</span>
    <span class="chip">KW R²&nbsp;=&nbsp;0.920</span>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
