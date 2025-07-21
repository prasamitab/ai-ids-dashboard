import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import plotly.express as px
from google.generativeai import configure, GenerativeModel

# Gemini API configuration
configure(api_key="YOUR_GEMINI_API_KEY")  # Replace with your Gemini API key
gemini = GenerativeModel("gemini-pro")

st.set_page_config(page_title="AI IDS", layout="centered", initial_sidebar_state="expanded")

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

chosen_theme = st.sidebar.radio(
    "Choose Theme", ["Light", "Dark"],
    index=0 if st.session_state.theme == "Light" else 1
)
st.session_state.theme = chosen_theme

light_style = "background-color: #f5f5f5; color: black;"
dark_style = "background-color: #1e1e1e; color: white;"
style = light_style if st.session_state.theme == "Light" else dark_style

st.markdown(f"""
<div style='{style} padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);'>
    <div style='display: flex; justify-content: center; align-items: center; margin-bottom: 10px;'>
        <img src="logo.png" alt="Logo" style="width: 100px; height: 60px; border-radius: 5px;">
    </div>
    <h1 style='text-align: center; color: {"black" if st.session_state.theme == "Light" else "white"};'> AI-Powered Intrusion Detection System 🔐</h1>
    <h4 style='text-align: center; color: {"black" if st.session_state.theme == "Light" else "white"};'>Detect cyber attacks in real-time using Machine Learning</h4>
    <p style='text-align: center; font-size: 14px; color: {"black" if st.session_state.theme == "Light" else "white"};'>Built by <b>Prasamita Bangal.</b> | Mahindra University</p>
</div>
""", unsafe_allow_html=True)

st.divider()

model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ["Random Forest", "Logistic Regression"]
)

@st.cache_resource
def load_resources(choice):
    model_file = "ids_model.pkl" if choice == "Random Forest" else "logistic_model.pkl"
    try:
        model = joblib.load(model_file)
        feature_list = joblib.load("model_features.pkl")
        return model, feature_list
    except FileNotFoundError:
        st.error(f"Model file '{model_file}' or 'model_features.pkl' not found.")
        st.stop()

model, feature_list = load_resources(model_choice)

with st.sidebar:
    st.title("About the App")
    st.markdown("""
    ## Navigation
    - Data Preview
    - Summary Metrics
    - Prediction Breakdown
    - Model Accuracy & Confusion Matrix
    - Top 10 Feature Importances
    - Live Streaming Simulation
    - Simulated Attack Map
    - Full Predictions (Top 25)
    """)
    st.markdown("""
    This app uses AI to detect cyber intrusions in real-time. Upload your data or use sample data to explore its features.
    """)
    st.divider()
    st.subheader("Feedback Form")
    with st.form("feedback_form"):
        feedback_name = st.text_input("Your Name")
        feedback_text = st.text_area("Your Feedback")
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            if feedback_name and feedback_text:
                st.success("Thank you for your feedback!")
                with open("feedback_log.txt", "a") as f:
                    f.write(f"{datetime.now()} - {feedback_name}: {feedback_text}\n")
            else:
                st.warning("Please fill in both fields.")

data = None
st.markdown("### Load Your Data")
col_sample, col_uploader = st.columns([0.4, 0.6])

with col_sample:
    if st.button("✨ Try with Sample Data"):
        try:
            data = pd.read_csv("preprocessed_test_sample_fixed.csv")
            st.session_state["sample_loaded"] = True
            st.success("Sample data loaded!")
        except FileNotFoundError:
            st.error("Sample data file not found.")
            st.session_state["sample_loaded"] = False

with col_uploader:
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.session_state["sample_loaded"] = True
        st.success("Data uploaded successfully!")

if st.session_state.get("sample_loaded") and data is not None:
    data = data.reindex(columns=feature_list, fill_value=0)

    st.divider()
    st.subheader("1. Data Preview :clipboard:")
    st.dataframe(data.head())

    predictions = model.predict(data)
    prediction_probs = model.predict_proba(data)
    labels = ["Normal" if i == 1 else "Attack" for i in predictions]
    confidences = [f"{max(p)*100:.2f}%" for p in prediction_probs]

    data["Prediction"] = labels
    data["Confidence"] = confidences

    total_attacks = data["Prediction"].value_counts().get("Attack", 0)
    total_normal = data["Prediction"].value_counts().get("Normal", 0)

    st.divider()
    st.subheader("2. Summary Metrics :bar_chart:")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Records", len(data))
    with col2: st.metric("Attacks Detected", total_attacks)
    with col3: st.metric("Normal Traffic", total_normal)
    with col4: st.metric("Attack %", f"{(total_attacks / len(data)) * 100:.2f}%")

    st.divider()
    st.subheader("3. Prediction Breakdown :chart_pie:")
    pred_counts = data["Prediction"].value_counts().reset_index()
    pred_counts.columns = ['Prediction', 'Count']
    fig_pred_breakdown = px.bar(pred_counts, x='Prediction', y='Count', color='Prediction')
    st.plotly_chart(fig_pred_breakdown, use_container_width=True)

    st.divider()
    st.subheader("4. Model Accuracy & Confusion Matrix :mag_right:")
    true_labels = [1 if lbl == "Normal" else 0 for lbl in labels]
    cm = confusion_matrix(true_labels, predictions)
    acc = accuracy_score(true_labels, predictions)
    st.write(f"**Accuracy Score:** {acc*100:.2f}%")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Attack", "Normal"], yticklabels=["Attack", "Normal"])
    st.pyplot(fig_cm)

    st.divider()
    st.subheader("5. Top 10 Feature Importances :key:")
    if hasattr(model, "feature_importances_"):
        feat_series = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False).head(10)
        st.bar_chart(feat_series)
    else:
        st.info("Feature importances not available for Logistic Regression.")

    st.divider()
    st.subheader("6. 📺 Live Streaming Simulation")
    if st.button("▶️ Start Stream Simulation"):
        import time
        live_placeholder = st.empty()
        progress_bar = st.progress(0)
        for i in range(min(25, len(data))):
            live_row = data.iloc[[i]][feature_list]
            pred = model.predict(live_row)[0]
            label = "Normal" if pred == 1 else "Attack"
            conf = f"{max(model.predict_proba(live_row)[0]) * 100:.2f}%"
            color = "green" if label == "Normal" else "red"
            live_placeholder.markdown(f"Row {i+1}: <span style='color:{color};'>{label}</span> (Confidence: {conf})", unsafe_allow_html=True)
            progress_bar.progress((i + 1) / 25)
            time.sleep(0.4)
        st.success("Simulation complete!")

    st.divider()
    st.subheader("7. Simulated Attack Map :world_map:")
    try:
        attack_data = data[data['Prediction'] == 'Attack'].copy()
        if not attack_data.empty:
            attack_data['lat'] = np.random.uniform(8.0, 37.0, len(attack_data))
            attack_data['lon'] = np.random.uniform(68.0, 97.0, len(attack_data))
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(latitude=22.0, longitude=78.0, zoom=3.5),
                layers=[
                    pdk.Layer('ScatterplotLayer', data=attack_data, get_position='[lon, lat]', get_color='[255, 0, 0, 160]', get_radius=40000)
                ],
                tooltip={"text": "Attack at {lon}, {lat}"}
            ))
        else:
            st.info("No attacks to show.")
    except Exception as e:
        st.error(f"Attack map error: {str(e)}")

    st.divider()
    st.subheader("8. 📄 Full Predictions (Top 25)")
    display_data = data.copy()
    display_data.insert(0, "🔒 Threat", data["Prediction"])
    st.dataframe(display_data.head(25))
    st.download_button("Download All Predictions", data=data.to_csv(index=False).encode("utf-8"), file_name="IDS_predictions.csv")

    st.divider()
    st.subheader("Ask the Assistant 💬")
    user_q = st.text_input("Have a question? (e.g., Why was row 3 flagged as Attack?)")
    if st.button("Ask Gemini"):
        if user_q.strip():
            st.info("Thinking...")
            try:
                reply = gemini.generate_content(user_q).text
                st.success("Gemini says:")
                st.write(reply)
            except Exception as e:
                st.error(f"Gemini API error: {str(e)}")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload data or use sample to begin.")
