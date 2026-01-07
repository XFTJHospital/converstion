# ====================================
# app.py —— Streamlit Web App
# ====================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 页面设置
st.set_page_config(page_title="rHCC AI Decision Model", layout="centered")

st.title("🧠 AI-based Decision Support for Ruptured HCC (rHCC)")
st.markdown("""
This app provides AI-assisted recommendations for optimal treatment strategy 
in ruptured hepatocellular carcinoma (**rHCC**).
""")

# =========================
# 1. 加载模型文件
# =========================
@st.cache_resource
def load_model():
    model_package = joblib.load("rHCC_model_merged.pkl")
    return (
        model_package["model"],
        model_package["feature_encoders"],
        model_package["target_encoder"],
        model_package["feature_names"]
    )

model, feature_encoders, target_encoder, feature_names = load_model()

# =========================
# 2. 用户输入
# =========================
st.subheader("🔍 Input Patient Information")

col1, col2 = st.columns(2)
with col1:
    tumor_length = st.number_input("Tumor max length (cm)", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
    AFP = st.number_input("AFP (ng/mL)", min_value=1, max_value=10000, value=300, step=10)
    age = st.selectbox("Age group", ["<60", "≥60"])
    child_pugh = st.selectbox("Child-Pugh grade", ["A", "B", "C"])

with col2:
    CSPH = st.selectbox("Clinically significant portal hypertension (CSPH)", ["No", "Yes"])
    hemo_instability = st.selectbox("Hemodynamic instability", ["No", "Yes"])
    protrusion = st.selectbox("Protrusion from liver surface", ["No", "Yes"])

# =========================
# 3. 数据预处理 + 预测
# =========================
if st.button("🔮 Predict Optimal Treatment"):
    input_data = pd.DataFrame({
        "Tumor_max_length": [tumor_length],
        "AFP": [AFP],
        "Age": [age],
        "Child_Pugh_grade": [child_pugh],
        "CSPH": [CSPH],
        "Hemodynamic_instability": [hemo_instability],
        "Protrusion_from_surface": [protrusion]
    })

    # 编码输入特征
    for col in feature_encoders.keys():
        le = feature_encoders[col]
        input_data[col] = le.transform(input_data[col])

    # 模型预测
    probs = model.predict_proba(input_data)[0]
    pred_label = model.predict(input_data)[0]
    treatment = target_encoder.inverse_transform([pred_label])[0]

    # =========================
    # 结果展示
    # =========================
    st.success(f"🏥 Recommended Treatment Strategy: **{treatment}**")

    st.write("### Probability for each treatment option:")
    prob_table = pd.DataFrame({
        "Treatment Strategy": target_encoder.classes_,
        "Predicted Probability": np.round(probs, 3)
    })
    st.table(prob_table)

    st.markdown("""
    ---
    **Interpretation:**  
    - Bridge TACE-to-surgery: Favorable for stable patients with good hepatic reserve  
    - Emergency surgery: For hemodynamically unstable patients  
    - TACE-only: For high-risk or poor hepatic reserve cases  
    """)

st.markdown("---")
st.caption("Developed by Tongji Hospital, Huazhong University of Science and Technology • Academic use only.")
