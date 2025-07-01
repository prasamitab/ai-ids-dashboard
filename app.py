import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

st.set_page_config(page_title="AI IDS", layout="centered")

# Theme toggle
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

chosen_theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
st.session_state.theme = chosen_theme

light_style = "background-color: #f5f5f5; color: black;"
dark_style = "background-color: #1e1e1e; color: white;"
style = light_style if st.session_state.theme == "Light" else dark_style
st.image("logo.png.jpeg", width=100)

# Styled Header
st.markdown(f"""
<div style='{style} padding: 10px;'>
<h1 style='text-align: center;'> AI-Powered Intrusion Detection System 🔐</h1>
<h4 style='text-align: center;'>Detect cyber attacks in real-time using Machine Learning</h4>
<p style='text-align: center; font-size: 14px;'>Built by <b>Prasamita Bangal.</b> | Mahindra University</p></div>
<hr style='border-top: 2px solid #bbb;'>
""", unsafe_allow_html=True)

# Model selector
model_choice = st.sidebar.selectbox("Choose a Model", ["Random Forest", "Logistic Regression"])

@st.cache_resource
def load_resources(choice):
    model_file = "ids_model.pkl" if choice == "Random Forest" else "logistic_model.pkl"
    model = joblib.load(model_file)
    feature_list = joblib.load("model_features.pkl")
    return model, feature_list

model, feature_list = load_resources(model_choice)

# Sidebar
with st.sidebar:
    st.title(" About the App")
    st.markdown("""
    ##  Navigation
    - 1] [Data Preview](#1️⃣-🔍-data-preview)
    - 2] [Summary Metrics](#2️⃣-📊-summary-metrics)
    - 3] [Prediction Breakdown](#3️⃣-📊-prediction-breakdown)
    - 4] [Accuracy & Confusion Matrix](#4️⃣-🧪-model-accuracy--confusion-matrix)
    - 5] [Top Features](#5️⃣-📌-top-10-feature-importances)
    - 6] [Streaming Simulation](#7️⃣-📺-live-streaming-simulation)
    - 7] [Attack Map](#8️⃣-🗺️-simulated-attack-map)
    - 8] [Full Predictions](#9️⃣-📄-full-predictions-top-25)

    """, unsafe_allow_html=True)
    st.markdown("""
    This is a lightweight, AI-powered intrusion detection dashboard built with:

    -  Random Forest classifier  
    -  NSL-KDD dataset  
    -  Real-time visualization  

    Try uploading a CSV or use sample data to test the IDS engine.
    """)

    st.markdown("---")
    st.subheader("📝 Feedback Form")
    feedback_name = st.text_input("Your Name")
    feedback_text = st.text_area("Your Feedback")
    if st.button("Submit Feedback"):
        st.success(" Thank you for your feedback!")
        with open("feedback_log.txt", "a") as f:
            f.write(f"{datetime.now()} - {feedback_name}: {feedback_text}\n")

# Sample data button
data = None
if st.button("✨ Try with Sample Data"):
    data = pd.read_csv("preprocessed_test_sample_fixed.csv")
    st.session_state["sample_loaded"] = True
else:
    uploaded_file = st.file_uploader(" Upload preprocessed_test_data.csv", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.session_state["sample_loaded"] = True

# Run predictions
if st.session_state.get("sample_loaded") and data is not None:
    data = data.reindex(columns=feature_list, fill_value=0)
    st.subheader(" 1. Data Preview")
    st.dataframe(data.head())

    predictions = model.predict(data)
    prediction_probs = model.predict_proba(data)
    labels = ["Normal" if i == 1 else "Attack" for i in predictions]
    confidences = [f"{max(p)*100:.2f}%" for p in prediction_probs]

    data["Prediction"] = labels
    data["Confidence"] = confidences

    total_attacks = data["Prediction"].value_counts().get("Attack", 0)
    total_normal = data["Prediction"].value_counts().get("Normal", 0)

    st.markdown("---")
    st.subheader(" 2.  Summary Metrics")
    st.metric("Total Records", len(data))
    st.metric("Attacks Detected", total_attacks)
    st.metric("Normal Traffic", total_normal)
    st.metric("Attack %", f"{(total_attacks / len(data)) * 100:.2f}%")

    # Additional summary as a visible table
    summary_df = pd.DataFrame({
        "Category": ["Total Records", "Attacks Detected", "Normal Traffic", "Attack %"],
        "Value": [len(data), total_attacks, total_normal, f"{(total_attacks / len(data)) * 100:.2f}%"]
    })
    st.dataframe(summary_df)

    st.markdown("---")
    st.subheader(" 3. Prediction Breakdown")
    pred_counts = data["Prediction"].value_counts()
    colors = ["#2ecc71" if label == "Normal" else "#e74c3c" for label in pred_counts.index]
    fig, ax = plt.subplots()
    bars = ax.bar(pred_counts.index, pred_counts.values, color=colors)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    ax.set_title("Attack vs Normal")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_facecolor('#f8f9fa')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    st.pyplot(fig)
    st.caption("🟢 Green = Normal   🔴 Red = Attack")

    st.markdown("---")
    st.subheader(" 4. Model Accuracy & Confusion Matrix")
    true_labels = [1 if lbl == "Normal" else 0 for lbl in labels]
    cm = confusion_matrix(true_labels, predictions)
    acc = accuracy_score(true_labels, predictions)

    st.write(f"**Accuracy Score:** {acc*100:.2f}% ✅")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Attack", "Normal"], yticklabels=["Attack", "Normal"], ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.markdown("---")
    st.subheader(" 5.  Top 10 Feature Importances")
    importances = model.feature_importances_
    feat_series = pd.Series(importances, index=feature_list).sort_values(ascending=False).head(10)
    st.bar_chart(feat_series)

    

    st.markdown("---")
    st.subheader(" 6.  📺 Live Streaming Simulation")
    if st.button("▶️ Start Stream Simulation"):
        import time
        live_placeholder = st.empty()
        for i in range(min(25, len(data))):
            live_row = data.iloc[[i]][feature_list]
            pred = model.predict(live_row)[0]
            label = "Normal" if pred == 1 else "Attack"
            conf = f"{max(model.predict_proba(live_row)[0]) * 100:.2f}%"
            live_placeholder.markdown(f"**Row {i+1}:** `{label}` (Confidence: {conf})")
            time.sleep(0.6)

    st.markdown("---")
    st.subheader(" 7. Simulated Attack Map")
    try:
        attack_data = data[data['Prediction'] == 'Attack'].copy()
        if not attack_data.empty:
            attack_data['lat'] = np.random.uniform(8.0, 37.0, len(attack_data))
            attack_data['lon'] = np.random.uniform(68.0, 97.0, len(attack_data))
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/dark-v9' if st.session_state.theme == 'Dark' else 'mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(latitude=22.0, longitude=78.0, zoom=3.5, pitch=0),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=attack_data,
                        get_position='[lon, lat]',
                        get_color='[255, 0, 0, 160]',
                        get_radius=40000,
                    )
                ]
            ))
        else:
            st.info("✅ No attacks detected to map.")
    except Exception as e:
        st.error("Could not generate attack map. Error: " + str(e))

    st.subheader(" 8.  Full Predictions (Top 25)")
    threat_emojis = ["🛡️" if lbl == "Normal" else "😈" for lbl in data["Prediction"]]
    display_data = data.copy()
    display_data.insert(0, "🔒 Threat", threat_emojis)
    st.dataframe(display_data.head(25))

    st.download_button(
        label="📥 Download Predictions CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="IDS_predictions.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("🔖 Generate PDF Report")
    import io
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    if st.button("📤 Generate PDF Summary"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        elements = [
            Image("logo.png", width=100, height=60),

            Spacer(1, 12),
            Paragraph("AI-Powered Intrusion Detection Report", styles['Title']),
            Spacer(1, 24),
            Paragraph("Generated by Prasamita B.", styles['Normal']),
            Paragraph("Mahindra University", styles['Normal']),
            Spacer(1, 12),
            Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']),
            Spacer(1, 24),
            Paragraph("This report summarizes the intrusion detection results for the uploaded network traffic logs using AI-based classifiers.", styles['Normal']),
            Spacer(1, 36)
        ]
        Spacer(1, 12),
        Paragraph(f"Model Used: {model_choice}", styles['Normal']),
        Spacer(1, 12),
        Paragraph(f"Total Records: {len(data)}", styles['Normal']),
        Paragraph(f"Attacks Detected: {total_attacks}", styles['Normal']),
        Paragraph(f"Normal Traffic: {total_normal}", styles['Normal']),
        Paragraph(f"Attack %: {(total_attacks / len(data)) * 100:.2f}%", styles['Normal']),
        Spacer(1, 12),
        Paragraph("Generated by Prasamita B.", styles['Normal']),
        Paragraph("Mahindra University", styles['Normal']),
        Spacer(1, 12),
        Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']),
        Spacer(1, 24),
        Paragraph("This report summarizes the intrusion detection results for the uploaded network traffic logs using AI-based classifiers.", styles['Normal']),
        Spacer(1, 36)

        ]
        doc.build(elements)
        st.download_button(
            label="📄 Download PDF Report",
            data=buffer.getvalue(),
            file_name="IDS_Report.pdf",
            mime="application/pdf"
        )

    st.markdown("---")
    with st.expander(" How This Works"):
        st.caption("🟢 = Normal Traffic  🔴 = Attack Traffic")
        st.caption(" Confidence = Model's certainty in its prediction")
        st.caption(" Streaming Simulation = Real-time row-by-row intrusion demo")
        st.markdown("""
        - Trained on the **NSL-KDD dataset**  
        - Features are one-hot encoded and standardized  
        - Model: Random Forest Classifier or Logistic Regression  
        - Predicts if traffic is **Attack** or **Normal**  
        - Includes confidence score and live streaming preview
        """)

    st.markdown("""
    ---
    <p style='text-align: center;'>🔒 Powered by Machine Learning | Streamlit App by <b>Prasamita Bangal.</b></p>
    """, unsafe_allow_html=True)
else:
    st.info("👆 Please upload your `preprocessed_test_data.csv` file or click 'Try with Sample Data' to see predictions.")

