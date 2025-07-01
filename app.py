import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI IDS", layout="centered")

# Styled Header
st.markdown("""
<h1 style='text-align: center; color: #FF4B4B;'>🔐 AI-Powered Intrusion Detection System</h1>
<h4 style='text-align: center;'>Detect cyber attacks in real-time using Machine Learning</h4>
<p style='text-align: center; font-size: 14px;'>Built by <b>Prasamita B.</b> | Mahindra University</p>
<hr style='border-top: 2px solid #bbb;'>
""", unsafe_allow_html=True)

# Load model and feature list
@st.cache_resource
def load_resources():
    model = joblib.load("ids_model.pkl")
    feature_list = joblib.load("model_features.pkl")
    return model, feature_list

model, feature_list = load_resources()

# Sidebar
with st.sidebar:
    st.title("📘 About the App")
    st.markdown("""
    This is a lightweight, AI-powered intrusion detection dashboard built with:

    - ✅ Random Forest classifier
    - 📚 NSL-KDD dataset
    - 📊 Real-time visualization

    Try uploading a CSV or use sample data to test the IDS engine.
    """)

# Sample data button
if st.button("✨ Try with Sample Data"):
    data = pd.read_csv("preprocessed_test_sample_fixed.csv")
    st.session_state["sample_loaded"] = True
else:
    uploaded_file = st.file_uploader("📁 Upload preprocessed_test_data.csv", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.session_state["sample_loaded"] = True

if st.session_state.get("sample_loaded"):
    data = data.reindex(columns=feature_list, fill_value=0)
    st.subheader("🔍 Data Preview")
    st.dataframe(data.head())

    # Predict
    predictions = model.predict(data)
    prediction_probs = model.predict_proba(data)
    labels = ["Normal" if i == 1 else "Attack" for i in predictions]
    confidences = [f"{max(p)*100:.2f}%" for p in prediction_probs]

    data['Prediction'] = labels
    data['Confidence'] = confidences

    # Metrics
    total_attacks = data['Prediction'].value_counts().get('Attack', 0)
    total_normal = data['Prediction'].value_counts().get('Normal', 0)

    st.markdown("---")
    st.subheader("📊 Summary Metrics")
    st.metric("Total Records", len(data))
    st.metric("Attacks Detected", total_attacks)
    st.metric("Normal Traffic", total_normal)
    st.metric("Attack %", f"{(total_attacks/len(data))*100:.2f}%")

    # Prediction bar chart
    st.markdown("---")
    st.subheader("📊 Prediction Breakdown")
    st.bar_chart(data['Prediction'].value_counts())

    # Feature importance
    st.markdown("---")
    st.subheader("📌 Top 10 Feature Importances")
    importances = model.feature_importances_
    feat_series = pd.Series(importances, index=feature_list).sort_values(ascending=False).head(10)
    st.bar_chart(feat_series)

    # Full result preview
    st.markdown("---")
    st.subheader("📄 Full Predictions (Top 25)")
    st.dataframe(data.head(25))

    # Download option
    st.download_button(
        label="📥 Download Predictions CSV",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name="IDS_predictions.csv",
        mime="text/csv"
    )

    # Expander: How it works
    st.markdown("---")
    with st.expander("🧠 How This Works"):
        st.markdown("""
        - This model is trained on the **NSL-KDD dataset**
        - One-hot encoded and scaled features
        - Model: Random Forest Classifier
        - Predicts whether each row of data is **Attack** or **Normal**
        - Includes confidence score and visual summary
        """)

    st.markdown("""
    ---
    <p style='text-align: center;'>🔒 Powered by Machine Learning | Streamlit App by <b>Prasamita B.</b></p>
    """, unsafe_allow_html=True)
else:
    st.info("👆 Please upload your `preprocessed_test_data.csv` file or click 'Try with Sample Data' to see predictions.")
