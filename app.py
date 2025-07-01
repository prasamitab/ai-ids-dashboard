import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
from datetime import datetime

st.set_page_config(page_title="AI IDS", layout="centered")

# Theme toggle
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

chosen_theme = st.sidebar.radio("🎨 Choose Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
st.session_state.theme = chosen_theme

light_style = "background-color: #f5f5f5; color: black;"
dark_style = "background-color: #1e1e1e; color: white;"
style = light_style if st.session_state.theme == "Light" else dark_style

# Styled Header
st.markdown(f"""
<div style='{style} padding: 10px;'>
<h1 style='text-align: center;'>🔐 AI-Powered Intrusion Detection System</h1>🔐 AI-Powered Intrusion Detection System</h1>
<h4 style='text-align: center;'>Detect cyber attacks in real-time using Machine Learning</h4>
<p style='text-align: center; font-size: 14px;'>Built by <b>Prasamita B.</b> | Mahindra University</p></div>
<hr style='border-top: 2px solid #bbb;'>
""", unsafe_allow_html=True)

# Load model and feature list
@st.cache_resource

def load_resources():
    model_choice = st.sidebar.selectbox("🧠 Choose a Model", ["Random Forest", "Logistic Regression"])
    model_file = "ids_model.pkl" if model_choice == "Random Forest" else "logistic_model.pkl"
    model = joblib.load(model_file)
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
    st.markdown("---")
    st.subheader("📝 Feedback Form")
    feedback_name = st.text_input("Your Name")
    feedback_text = st.text_area("Your Feedback")
    if st.button("Submit Feedback"):
        st.success("✅ Thank you for your feedback!")
        with open("feedback_log.txt", "a") as f:
            f.write(f"{datetime.now()} - {feedback_name}: {feedback_text}\n")

# Raw log upload and auto-preprocessing
raw_uploaded = st.file_uploader("📤 Upload raw NSL-KDD data for auto-preprocessing (optional)", type="csv")
if raw_uploaded is not None:
    raw_data = pd.read_csv(raw_uploaded)
    st.write("Auto-preprocessing raw data...")

    if 'label' in raw_data.columns:
        raw_data.drop('label', axis=1, inplace=True)

    raw_data = pd.get_dummies(raw_data)
    raw_data = raw_data.reindex(columns=feature_list, fill_value=0)
    raw_data = (raw_data - raw_data.mean()) / raw_data.std()
    raw_data.to_csv("auto_preprocessed_test_data.csv", index=False)
    st.success("✅ Raw log preprocessed successfully!")
    data = raw_data
    st.session_state["sample_loaded"] = True

# Sample data or upload
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
    st.markdown(f"<div style='{style} padding:10px'>", unsafe_allow_html=True)
    st.subheader("🔍 Data Preview")
    st.dataframe(data.head())

    predictions = model.predict(data)
    prediction_probs = model.predict_proba(data)
    labels = ["Normal" if i == 1 else "Attack" for i in predictions]
    confidences = [f"{max(p)*100:.2f}%" for p in prediction_probs]

    data['Prediction'] = labels
    data['Confidence'] = confidences

    total_attacks = data['Prediction'].value_counts().get('Attack', 0)
    total_normal = data['Prediction'].value_counts().get('Normal', 0)

    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<div style='{style} padding:10px'>", unsafe_allow_html=True)
st.subheader("📊 Summary Metrics")
st.metric("Total Records", len(data))
st.metric("Attacks Detected", total_attacks)
st.metric("Normal Traffic", total_normal)
st.metric("Attack %", f"{(total_attacks/len(data))*100:.2f}%")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<div style='{style} padding:10px'>", unsafe_allow_html=True)
st.subheader("🗺️ Simulated Attack Map")
attack_data = data[data['Prediction'] == 'Attack'].copy()
if not attack_data.empty:
        attack_data['lat'] = np.random.uniform(8, 37, len(attack_data))
        attack_data['lon'] = np.random.uniform(68, 97, len(attack_data))
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(latitude=22, longitude=78, zoom=3.5),
            layers=[
                pdk.Layer('ScatterplotLayer', data=attack_data,
                          get_position='[lon, lat]',
                          get_color='[200, 30, 0, 160]',
                          get_radius=20000)
            ]))
    else:
        st.info("✅ No attacks detected, map not shown.")

    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<div style='{style} padding:10px'>", unsafe_allow_html=True)
st.subheader("📊 Prediction Breakdown")
    fig, ax = plt.subplots()
    ax.bar(data['Prediction'].value_counts(.index, data['Prediction'].value_counts(.values, color='white' if st.session_state.theme == 'Dark' else '#FF4B4B')
    ax.set_facecolor('#1e1e1e' if st.session_state.theme == 'Dark' else 'white')
    ax.tick_params(colors='white' if st.session_state.theme == 'Dark' else 'black')
    st.pyplot(fig))

    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<div style='{style} padding:10px'>", unsafe_allow_html=True)
st.subheader("📌 Top 10 Feature Importances")
    importances = model.feature_importances_
    feat_series = pd.Series(importances, index=feature_list).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    ax.bar(feat_series.index, feat_series.values, color='white' if st.session_state.theme == 'Dark' else '#FF4B4B')
    ax.set_facecolor('#1e1e1e' if st.session_state.theme == 'Dark' else 'white')
    ax.tick_params(colors='white' if st.session_state.theme == 'Dark' else 'black')
    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<div style='{style} padding:10px'>", unsafe_allow_html=True)
st.subheader("📺 Live Streaming Simulation")
    if st.button("▶️ Start Stream Simulation"):
        import time
        live_placeholder = st.empty()
        for i in range(min(25, len(data))):
            live_row = data.iloc[[i]][feature_list]
            pred = model.predict(live_row)[0]
            label = "Normal" if pred == 1 else "Attack"
            conf = f"{max(model.predict_proba(live_row)[0])*100:.2f}%"
            live_placeholder.markdown(f"**Row {i+1}:** `{label}` (Confidence: {conf})")
            time.sleep(0.6)

    

    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<div style='{style} padding:10px'>", unsafe_allow_html=True)
st.subheader("🔍 Explain a Prediction with SHAP")
    shap_explainer = joblib.load("shap_explainer.pkl")
    row_index = st.number_input("Choose a row index to explain (0–49 recommended):", min_value=0, max_value=min(len(data)-1, 49), value=0)
    selected_row = data.iloc[[row_index]][feature_list]
    shap_values = shap_explainer(selected_row)
    st.write("Feature impact on this prediction:")
    shap_df = pd.DataFrame({
        "Feature": feature_list,
        "SHAP Value": shap_values.values[0]
    }).sort_values("SHAP Value", key=abs, ascending=False).head(10)
    fig, ax = plt.subplots()
    ax.bar(shap_df.set_index("Feature".index, shap_df.set_index("Feature".values, color='white' if st.session_state.theme == 'Dark' else '#FF4B4B')
    ax.set_facecolor('#1e1e1e' if st.session_state.theme == 'Dark' else 'white')
    ax.tick_params(colors='white' if st.session_state.theme == 'Dark' else 'black')
    st.pyplot(fig))

    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<div style='{style} padding:10px'>", unsafe_allow_html=True)
st.subheader("📄 Full Predictions (Top 25)") (Top 25)")
    st.dataframe(data.head(25))

    st.download_button(
        label="📥 Download Predictions CSV",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name="IDS_predictions.csv",
        mime="text/csv"
    )

    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"<div style='{style} padding:10px'>", unsafe_allow_html=True)
st.subheader("🧾 Generate PDF Report (Coming Soon)")
    st.info("This feature will allow you to export a PDF summary of your analysis. Stay tuned!")

    st.markdown("---")
    with st.expander("🧠 How This Works"):
        st.markdown("""
        - Trained on the **NSL-KDD dataset**
        - One-hot encoded and standardized features
        - Choose between **Random Forest** or **Logistic Regression**
        - Real-time confidence score, bar charts, and simulated map
        - Auto-preprocessing for raw logs
        """)

    st.markdown("""
    ---
    <p style='text-align: center;'>🔒 Powered by Machine Learning | Streamlit App by <b>Prasamita B.</b></p>
    """, unsafe_allow_html=True)
else:
    st.info("👆 Please upload your `preprocessed_test_data.csv` file or click 'Try with Sample Data' to see predictions.")
