import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fetal Health Predictor",
    page_icon="🩺",
    layout="wide"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { max-width: 1200px; margin: auto; }
    .result-box {
        padding: 20px; border-radius: 12px;
        text-align: center; font-size: 20px;
        font-weight: bold; margin-top: 20px;
    }
    .normal   { background-color: #d4edda; color: #155724; border: 2px solid #28a745; }
    .suspect  { background-color: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .patho    { background-color: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
    .prob-bar { height: 28px; border-radius: 6px; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADING (no upload needed)
# ─────────────────────────────────────────────
MODEL_PATH    = "fetal_model.joblib"
SCALER_PATH   = "fetal_scaler.joblib"
FEATURES_PATH = "fetal_features.joblib"

@st.cache_resource
def load_model():
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

try:
    model, scaler, features = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR — INFO ONLY
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("This app uses a pre-trained **Gradient Boosting** model trained on Cardiotocogram (CTG) data.")
    st.success("✅ Model loaded and ready!")
    st.info(f"Using **{len(features)}** selected features")
    st.markdown("---")
    st.markdown("**Target Classes:**")
    st.markdown("🟢 **1 – Normal**")
    st.markdown("🟡 **2 – Suspect**")
    st.markdown("🔴 **3 – Pathological**")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🩺 Fetal Health Prediction App")
st.markdown("""
> This tool uses a **Gradient Boosting** machine learning model trained on Cardiotocogram (CTG) 
> data to predict fetal health status. Predictions are presented as probabilities to support 
> clinical decision-making.
""")
st.divider()

# ─────────────────────────────────────────────
# MAIN — INPUT FORM
# ─────────────────────────────────────────────
st.subheader("📋 Enter Patient CTG Readings")
st.markdown("Adjust the sliders below to match the patient's CTG measurements.")

col1, col2 = st.columns(2)

with col1:
    baseline_value = st.slider(
        "Baseline Fetal Heart Rate (bpm)",
        min_value=100, max_value=180, value=133,
        help="Average fetal heart rate at rest"
    )
    accelerations = st.slider(
        "Accelerations (per second)",
        min_value=0.000, max_value=0.020, value=0.003, step=0.001, format="%.3f",
        help="Short-term increases in heart rate — a healthy sign"
    )
    abnormal_stv = st.slider(
        "Abnormal Short-Term Variability (%)",
        min_value=0, max_value=100, value=20,
        help="% of time with abnormal rapid heart rate changes"
    )
    mean_stv = st.slider(
        "Mean Short-Term Variability",
        min_value=0.0, max_value=10.0, value=1.5, step=0.1,
        help="Average rapid fluctuations in heart rate"
    )
    pct_abnormal_ltv = st.slider(
        "% Time with Abnormal Long-Term Variability",
        min_value=0, max_value=100, value=0,
        help="Higher values can indicate fetal problems"
    )

with col2:
    mean_ltv = st.slider(
        "Mean Long-Term Variability",
        min_value=0.0, max_value=50.0, value=8.0, step=0.5,
        help="Average long-term heart rate changes"
    )
    histogram_width = st.slider(
        "Histogram Width",
        min_value=0, max_value=200, value=70,
        help="Range of heart rate values observed"
    )
    histogram_mode = st.slider(
        "Histogram Mode",
        min_value=60, max_value=190, value=137,
        help="Most common heart rate value"
    )
    histogram_mean = st.slider(
        "Histogram Mean",
        min_value=60, max_value=190, value=136,
        help="Average heart rate value"
    )
    histogram_median = st.slider(
        "Histogram Median",
        min_value=60, max_value=190, value=138,
        help="Median heart rate value"
    )

# ─────────────────────────────────────────────
# BUILD INPUT DATAFRAME
# ─────────────────────────────────────────────
input_map = {
    'baseline value':                                         baseline_value,
    'accelerations':                                          accelerations,
    'abnormal_short_term_variability':                        abnormal_stv,
    'mean_value_of_short_term_variability':                   mean_stv,
    'percentage_of_time_with_abnormal_long_term_variability': pct_abnormal_ltv,
    'mean_value_of_long_term_variability':                    mean_ltv,
    'histogram_width':                                        histogram_width,
    'histogram_mode':                                         histogram_mode,
    'histogram_mean':                                         histogram_mean,
    'histogram_median':                                       histogram_median,
}

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔍 Predict Fetal Health", type="primary", use_container_width=True)

if predict_btn:
    all_feature_names = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else features
    row = {f: 0.0 for f in all_feature_names}
    row.update(input_map)

    input_df_full = pd.DataFrame([row])
    try:
        input_scaled = pd.DataFrame(
            scaler.transform(input_df_full),
            columns=all_feature_names
        )
    except Exception:
        input_scaled = pd.DataFrame(
            scaler.transform(input_df_full[features]),
            columns=features
        )

    input_selected = input_scaled[features]

    probabilities   = model.predict_proba(input_selected)[0]
    predicted_class = model.predict(input_selected)[0]

    class_labels = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    class_styles = {1: "normal", 2: "suspect", 3: "patho"}
    class_icons  = {1: "🟢", 2: "🟡", 3: "🔴"}

    pred_label = class_labels[predicted_class]
    pred_style = class_styles[predicted_class]
    pred_icon  = class_icons[predicted_class]
    pred_prob  = probabilities[predicted_class - 1] * 100

    st.subheader("📊 Prediction Results")

    st.markdown(f"""
    <div class="result-box {pred_style}">
        {pred_icon} The model predicts a <u>{pred_prob:.2f}%</u> probability that this patient's 
        fetal health is <strong>{pred_label}</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Probability Breakdown")

    colors = {1: "#28a745", 2: "#ffc107", 3: "#dc3545"}
    for i, (cls, label) in enumerate({1: "Normal", 2: "Suspect", 3: "Pathological"}.items()):
        prob_pct  = probabilities[i] * 100
        bar_color = colors[cls]
        icon      = class_icons[cls]
        st.markdown(f"**{icon} {label}**")
        st.markdown(f"""
        <div style="background:#e9ecef; border-radius:6px; height:28px; width:100%;">
            <div style="width:{prob_pct:.1f}%; background:{bar_color}; height:28px; 
                        border-radius:6px; display:flex; align-items:center; 
                        padding-left:8px; color:white; font-weight:bold; font-size:13px;">
                {prob_pct:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

    st.markdown("#### 💡 Clinical Guidance")
    guidance = {
        1: "✅ **Normal**: Fetal condition appears healthy. Continue routine monitoring and antenatal care.",
        2: "⚠️ **Suspect**: Borderline readings detected. Consider additional monitoring, repeat CTG, and clinical review.",
        3: "🚨 **Pathological**: High-risk signs detected. Immediate clinical intervention and specialist review is strongly advised."
    }
    st.info(guidance[predicted_class])

    st.caption("⚠️ This tool is a decision-support aid only. All clinical decisions must be made by a qualified healthcare professional.")