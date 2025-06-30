import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI IDS", layout="centered")
st.title("🔐 AI-Powered Intrusion Detection System")
st.markdown("""
Upload preprocessed network log data and detect cyber attacks using AI.
Built by **Prasamita B.**, Mahindra University.
""")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("ids_model.pkl")

model = load_model()

# Upload CSV file
uploaded_file = st.file_uploader("📁 Upload preprocessed_test_data.csv", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(data.head())

    # Predict
    predictions = model.predict(data)
    prediction_probs = model.predict_proba(data)
    labels = ["Normal" if i == 1 else "Attack" for i in predictions]
    confidences = [f"{max(p)*100:.2f}%" for p in prediction_probs]

    data['Prediction'] = labels
    data['Confidence'] = confidences

    # Metrics summary
    total_attacks = data['Prediction'].value_counts().get('Attack', 0)
    total_normal = data['Prediction'].value_counts().get('Normal', 0)

    st.subheader("📊 Summary Metrics")
    st.metric("Total Records", len(data))
    st.metric("Attacks Detected", total_attacks)
    st.metric("Normal Traffic", total_normal)

    # Bar chart
    st.subheader("📊 Prediction Breakdown")
    st.bar_chart(data['Prediction'].value_counts())

    # Feature importance chart
    st.subheader("📌 Model Feature Importance")
    importances = model.feature_importances_
    feat_series = pd.Series(importances, index=data.columns[:-2]).sort_values(ascending=False).head(10)
    st.bar_chart(feat_series)

    # Full result preview
    st.subheader("📄 Full Predictions (Top 25)")
    st.dataframe(data.head(25))

    # Download option
    st.download_button(
        label="📥 Download Predictions CSV",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name="IDS_predictions.csv",
        mime="text/csv"
    )

    # About section
    st.markdown("""
    ---
    **About:** This app uses a trained Random Forest model on the NSL-KDD dataset to classify network traffic.
    Developed and deployed by *Prasamita B.*
    """)
else:
    st.info("👆 Please upload your `preprocessed_test_data.csv` file to see predictions.")
