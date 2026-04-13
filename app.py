import streamlit as st
import numpy as np
import joblib
import os

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioCheck — Heart Risk Assessment",
    layout="centered",
    page_icon="❤️"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main {
    background: #fafaf8;
}

.block-container {
    max-width: 720px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(231,76,60,0.3) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: rgba(255,255,255,0.65);
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}
.hero-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
}

/* Section card */
.section-card {
    background: white;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid #ebebeb;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #1a1a2e;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #f0f0f0;
}

/* Risk result */
.risk-high {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    color: white;
}
.risk-low {
    background: linear-gradient(135deg, #1e8449, #27ae60);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    color: white;
}
.risk-medium {
    background: linear-gradient(135deg, #d35400, #e67e22);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    color: white;
}
.risk-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: 0.5rem 0;
}
.risk-score {
    font-size: 3.5rem;
    font-weight: 600;
    margin: 0;
}
.risk-icon { font-size: 2.5rem; }
.risk-desc {
    font-size: 0.9rem;
    opacity: 0.88;
    margin-top: 0.8rem;
    line-height: 1.5;
}

/* Disclaimer */
.disclaimer {
    background: #fff8e1;
    border-left: 4px solid #f39c12;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: #7d6608;
    margin-top: 1.5rem;
}

/* Model missing */
.model-warning {
    background: #fdecea;
    border-left: 4px solid #e74c3c;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: #c0392b;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">❤️</span>
    <p class="hero-title">CardioCheck</p>
    <p class="hero-subtitle">Answer a few questions to receive your personalised cardiovascular risk assessment</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH = "cardio_logistic_model.pkl"

uploaded_model = st.sidebar.file_uploader("📦 Upload your trained model (.pkl)", type=["pkl"])
if uploaded_model:
    with open(MODEL_PATH, "wb") as f:
        f.write(uploaded_model.read())
    st.sidebar.success("Model loaded!")

if not os.path.exists(MODEL_PATH):
    st.markdown("""
    <div class="model-warning">
        <strong>⚠️ No model found.</strong><br>
        Please upload your <code>cardio_logistic_model.pkl</code> file using the sidebar uploader.
        Train it first using the <strong>training app</strong>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

pipeline = joblib.load(MODEL_PATH)

# ─── Form ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
with col2:
    weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
    gender = st.selectbox("Gender", ["Female (1)", "Male (2)"])
    gender_val = 1 if "Female" in gender else 2

bmi = weight / ((height / 100) ** 2)
st.markdown(f"📐 Your BMI: **{bmi:.1f}**")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">🩺 Blood Pressure & Cholesterol</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    ap_hi = st.number_input("Systolic BP (ap_hi) mmHg", min_value=60, max_value=250, value=120)
    cholesterol = st.selectbox("Cholesterol Level", ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"])
    chol_val = int(cholesterol.split("(")[1].replace(")", ""))
with col2:
    ap_lo = st.number_input("Diastolic BP (ap_lo) mmHg", min_value=40, max_value=180, value=80)
    gluc = st.selectbox("Glucose Level", ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"])
    gluc_val = int(gluc.split("(")[1].replace(")", ""))

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">🚬 Lifestyle</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    smoke = st.radio("Smoker?", ["No", "Yes"])
    smoke_val = 1 if smoke == "Yes" else 0
with col2:
    alco = st.radio("Drinks Alcohol?", ["No", "Yes"])
    alco_val = 1 if alco == "Yes" else 0
with col3:
    active = st.radio("Physically Active?", ["Yes", "No"])
    active_val = 1 if active == "Yes" else 0

st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict ─────────────────────────────────────────────────────────────────
if st.button("🔍 Check My Heart Risk", use_container_width=True, type="primary"):

    # Build input — must match training feature order
    # Features: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, BMI
    input_data = np.array([[age, gender_val, height, weight, ap_hi, ap_lo,
                            chol_val, gluc_val, smoke_val, alco_val, active_val, bmi]])

    # Try to match the exact feature set the model was trained on
    try:
        prob = pipeline.predict_proba(input_data)[0][1]
        risk_pct = round(prob * 100, 1)

        if risk_pct >= 60:
            css_class = "risk-high"
            icon = "🚨"
            label = "High Risk"
            desc = "Your responses suggest a higher likelihood of cardiovascular disease. Please consult a doctor as soon as possible for a professional evaluation."
        elif risk_pct >= 40:
            css_class = "risk-medium"
            icon = "⚠️"
            label = "Moderate Risk"
            desc = "There are some risk factors present. Consider lifestyle changes and speak with a healthcare provider for guidance."
        else:
            css_class = "risk-low"
            icon = "✅"
            label = "Low Risk"
            desc = "Your responses suggest a lower risk of cardiovascular disease. Keep maintaining a healthy lifestyle!"

        st.markdown(f"""
        <div class="{css_class}">
            <div class="risk-icon">{icon}</div>
            <div class="risk-label">{label}</div>
            <div class="risk-score">{risk_pct}%</div>
            <div class="risk-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        # Factor summary
        st.markdown("---")
        st.markdown("#### 📋 Your Risk Factor Summary")
        factors = {
            "🩸 Blood Pressure": f"{ap_hi}/{ap_lo} mmHg — {'⚠️ Elevated' if ap_hi > 130 else '✅ Normal'}",
            "⚖️ BMI": f"{bmi:.1f} — {'⚠️ Overweight/Obese' if bmi >= 25 else '✅ Healthy'}",
            "🧪 Cholesterol": cholesterol,
            "🍬 Glucose": gluc,
            "🚬 Smoking": f"{'⚠️ Yes' if smoke_val else '✅ No'}",
            "🏃 Physical Activity": f"{'✅ Active' if active_val else '⚠️ Inactive'}",
        }
        for k, v in factors.items():
            st.markdown(f"**{k}**: {v}")

    except Exception as e:
        st.error(f"Prediction failed: {e}. Make sure the model was trained with the same features.")

# ─── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚕️ <strong>Medical Disclaimer:</strong> This tool is for educational purposes only and is not a substitute
    for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider
    with any questions about your health.
</div>
""", unsafe_allow_html=True)
