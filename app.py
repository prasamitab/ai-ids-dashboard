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
import plotly.express as px # Import Plotly for interactive charts

st.set_page_config(page_title="AI IDS", layout="centered", initial_sidebar_state="expanded")

# --- Theme Toggle ---
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

chosen_theme = st.sidebar.radio(
    "Choose Theme", ["Light", "Dark"],
    index=0 if st.session_state.theme == "Light" else 1,
    help="Switch between light and dark mode for the application."
)
st.session_state.theme = chosen_theme

light_style = "background-color: #f5f5f5; color: black;"
dark_style = "background-color: #1e1e1e; color: white;"
style = light_style if st.session_state.theme == "Light" else dark_style

# --- Header Section ---
# Using a container for a more defined header area
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
st.divider() # Replaces <hr> for a cleaner look

# --- Model Selector ---
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ["Random Forest", "Logistic Regression"],
    help="Select the machine learning model to use for intrusion detection. Random Forest is generally more accurate, Logistic Regression is faster."
)

@st.cache_resource
def load_resources(choice):
    """
    Loads the pre-trained machine learning model and feature list.
    Uses st.cache_resource to avoid reloading on every rerun.
    """
    model_file = "ids_model.pkl" if choice == "Random Forest" else "logistic_model.pkl"
    # In a real application, ensure these files are accessible or provided by the user.
    # For this example, we assume they exist in the same directory.
    try:
        model = joblib.load(model_file)
        feature_list = joblib.load("model_features.pkl")
        return model, feature_list
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_file}' or 'model_features.pkl' not found. Please ensure they are in the same directory as the app.py file.")
        st.stop() # Stop execution if files are missing

model, feature_list = load_resources(model_choice)

# --- Sidebar Content ---
with st.sidebar:
    st.title("About the App")
    st.markdown("""
    ## Navigation
    - 1. Data Preview
    - 2. Summary Metrics
    - 3. Prediction Breakdown
    - 4. Model Accuracy & Confusion Matrix
    - 5. Top 10 Feature Importances
    - 6. Live Streaming Simulation
    - 7. Simulated Attack Map
    - 8. Full Predictions (Top 25)
    """)
    st.markdown("""
    This is a lightweight, AI-powered intrusion detection dashboard built with:
    - **Random Forest** or **Logistic Regression** models
    - Trained on the **NSL-KDD dataset**
    - Features real-time visualization and interactive components.
    Try uploading your own preprocessed CSV data or use the sample data to test the IDS engine.
    """)

    st.divider()
    st.subheader("Feedback Form")
    with st.form("feedback_form"):
        feedback_name = st.text_input("Your Name", key="feedback_name")
        feedback_text = st.text_area("Your Feedback", key="feedback_text")
        submitted = st.form_submit_button("Submit Feedback :email:")
        if submitted:
            if feedback_name and feedback_text:
                st.success("Thank you for your feedback! We appreciate it.")
                # In a real app, you would save this to a database or a more robust logging system.
                with open("feedback_log.txt", "a") as f:
                    f.write(f"{datetime.now()} - {feedback_name}: {feedback_text}\n")
            else:
                st.warning("Please fill in both your name and feedback before submitting.")

# --- Data Loading Section ---
data = None
st.markdown("### Load Your Data")
col_sample, col_uploader = st.columns([0.4, 0.6])

with col_sample:
    if st.button("✨ Try with Sample Data", help="Load a pre-defined sample dataset to quickly see the app in action."):
        with st.spinner('Loading sample data...'):
            # Assuming 'preprocessed_test_sample_fixed.csv' exists
            try:
                data = pd.read_csv("preprocessed_test_sample_fixed.csv")
                st.session_state["sample_loaded"] = True
                st.success("Sample data loaded successfully!")
            except FileNotFoundError:
                st.error("Sample data file 'preprocessed_test_sample_fixed.csv' not found. Please ensure it's in the same directory.")
                st.session_state["sample_loaded"] = False
                data = None

with col_uploader:
    uploaded_file = st.file_uploader("Upload your `preprocessed_test_data.csv`", type="csv", help="Upload your own preprocessed network traffic data in CSV format.")
    if uploaded_file:
        with st.spinner('Processing uploaded file...'):
            data = pd.read_csv(uploaded_file)
            st.session_state["sample_loaded"] = True # Use this flag for both sample and uploaded
            st.success("File uploaded and processed!")

# --- Main Application Logic (after data is loaded) ---
if st.session_state.get("sample_loaded") and data is not None:
    # Ensure all required features are present, fill missing with 0
    data = data.reindex(columns=feature_list, fill_value=0)

    st.divider()
    st.subheader("1. Data Preview :clipboard:")
    st.dataframe(data.head()) # Displaying first few rows of the data

    # Perform predictions
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
    # Using columns for a cleaner display of metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(data), help="Total number of network traffic records analyzed.")
    with col2:
        st.metric("Attacks Detected", total_attacks, help="Number of records classified as attack traffic.")
    with col3:
        st.metric("Normal Traffic", total_normal, help="Number of records classified as normal traffic.")
    with col4:
        st.metric("Attack %", f"{(total_attacks / len(data)) * 100:.2f}%", help="Percentage of total traffic classified as attack.")

    st.divider()
    st.subheader("3. Prediction Breakdown :chart_pie:")
    pred_counts = data["Prediction"].value_counts().reset_index()
    pred_counts.columns = ['Prediction', 'Count']

    # Using Plotly for an interactive bar chart
    fig_pred_breakdown = px.bar(
        pred_counts,
        x='Prediction',
        y='Count',
        color='Prediction',
        color_discrete_map={"Normal": "#2ecc71", "Attack": "#e74c3c"}, # Green for Normal, Red for Attack
        title="Distribution of Attack vs Normal Traffic",
        labels={'Prediction': 'Traffic Type', 'Count': 'Number of Records'},
        text='Count' # Display count on bars
    )
    fig_pred_breakdown.update_layout(xaxis_title_text='Traffic Type', yaxis_title_text='Count')
    st.plotly_chart(fig_pred_breakdown, use_container_width=True)
    st.caption("🟢 Green = Normal Traffic  🔴 Red = Attack Traffic")


    st.divider()
    st.subheader("4. Model Accuracy & Confusion Matrix :mag_right:")
    # Convert labels back to numerical for accuracy/confusion matrix calculation if needed
    # Assuming 1 for Normal, 0 for Attack based on initial prediction logic
    true_labels_for_metrics = [1 if lbl == "Normal" else 0 for lbl in labels]
    cm = confusion_matrix(true_labels_for_metrics, predictions)
    acc = accuracy_score(true_labels_for_metrics, predictions)

    st.write(f"**Accuracy Score:** <span style='font-size: 1.2em; color: #28a745;'>**{acc*100:.2f}%**</span>", unsafe_allow_html=True)

    # Plotting Confusion Matrix with Seaborn
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Attack", "Normal"],
                yticklabels=["Attack", "Normal"], ax=ax_cm,
                linewidths=.5, linecolor='black')
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)
    st.caption("The confusion matrix shows the counts of correct and incorrect predictions.")

    st.divider()
    st.subheader("5. Top 10 Feature Importances :key:")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_series = pd.Series(importances, index=feature_list).sort_values(ascending=False).head(10)
        st.bar_chart(feat_series)
        st.caption("Features with higher importance contribute more to the model's prediction.")
    else:
        st.info("Feature importances are not available for the selected model (Logistic Regression). This metric is typically available for tree-based models like Random Forest.")

    st.divider()
    st.subheader("6. 📺 Live Streaming Simulation")
    st.info("Watch a real-time row-by-row simulation of intrusion detection.")
    if st.button("▶️ Start Stream Simulation", help="Simulate real-time detection for the first 25 records."):
        import time
        live_placeholder = st.empty()
        progress_bar = st.progress(0)
        total_rows_to_simulate = min(25, len(data))
        st.write(f"Simulating {total_rows_to_simulate} records...")

        for i in range(total_rows_to_simulate):
            live_row = data.iloc[[i]][feature_list]
            pred = model.predict(live_row)[0]
            label = "Normal" if pred == 1 else "Attack"
            conf = f"{max(model.predict_proba(live_row)[0]) * 100:.2f}%"

            color = "green" if label == "Normal" else "red"
            live_placeholder.markdown(f"**Row {i+1}:** <span style='color:{color};'>`{label}`</span> (Confidence: {conf})", unsafe_allow_html=True)
            progress_bar.progress((i + 1) / total_rows_to_simulate)
            time.sleep(0.6)
        st.success("Stream simulation complete! 🎉")

    st.divider()
    st.subheader("7. Simulated Attack Map :world_map:")
    st.info("Visualize detected attacks on a simulated world map. (Coordinates are random for demonstration).")
    try:
        attack_data = data[data['Prediction'] == 'Attack'].copy()
        if not attack_data.empty:
            # Assign random coordinates within a reasonable range for India for demonstration
            attack_data['lat'] = np.random.uniform(8.0, 37.0, len(attack_data))
            attack_data['lon'] = np.random.uniform(68.0, 97.0, len(attack_data))
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/dark-v9' if st.session_state.theme == 'Dark' else 'mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(latitude=22.0, longitude=78.0, zoom=3.5, bearing=0, pitch=45),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=attack_data,
                        get_position='[lon, lat]',
                        get_color='[255, 0, 0, 160]', # Red color for attacks
                        get_radius=40000, # Radius in meters
                        pickable=True,
                        auto_highlight=True
                    )
                ],
                tooltip={"text": "Attack detected at {lon}, {lat}"}
            ))
        else:
            st.info("No attacks detected in the dataset to map.")
    except Exception as e:
        st.error(f"Could not generate attack map: {str(e)}. Ensure you have an active internet connection for map tiles.")

    st.divider()
    st.subheader("8. 📄 Full Predictions (Top 25)")
    st.info("Review the detailed predictions for the first 25 records.")
    display_data = data.copy()
    # Insert 'Threat' column at the beginning for better visibility
    display_data.insert(0, "🔒 Threat", data["Prediction"])
    st.dataframe(display_data.head(25))

    st.download_button(
        label="Download All Predictions as CSV :arrow_down:",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="IDS_predictions.csv",
        mime="text/csv",
        help="Download the complete dataset with predictions and confidence scores."
    )

    st.divider()
    st.subheader("🔖 Generate PDF Report")
    if st.button("📤 Generate PDF Summary :page_facing_up:", help="Create a PDF report summarizing the intrusion detection results."):
        with st.spinner("Generating PDF report..."):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()
            elements = [
                # Using the local 'logo.png' file for the PDF report
                Image("logo.png", width=100, height=60),
                Spacer(1, 12),
                Paragraph("AI-Powered Intrusion Detection Report", styles['Title']),
                Spacer(1, 24),
                Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']),
                Spacer(1, 12),
                Paragraph(f"Model Used: {model_choice}", styles['Normal']),
                Paragraph(f"Total Records: {len(data)}", styles['Normal']),
                Paragraph(f"Attacks Detected: {total_attacks}", styles['Normal']),
                Paragraph(f"Normal Traffic: {total_normal}", styles['Normal']),
                Paragraph(f"Attack %: {(total_attacks / len(data)) * 100:.2f}%", styles['Normal']),
                Spacer(1, 24),
                Paragraph("This report summarizes the intrusion detection results for the uploaded network traffic logs using AI-based classifiers.", styles['Normal']),
                Spacer(1, 36),
                Paragraph("Generated by Prasamita Bangal.", styles['Normal']),
                Paragraph("Mahindra University", styles['Normal']),
            ]
            doc.build(elements)
            st.download_button(
                label="Download PDF Report",
                data=buffer.getvalue(),
                file_name="IDS_Report.pdf",
                mime="application/pdf"
            )
            st.success("PDF report generated successfully! Check your downloads.")

    st.divider()
    with st.expander("How This Works :bulb:"):
        st.caption("🟢 Green = Normal Traffic  🔴 Red = Attack Traffic")
        st.caption("Confidence = Model's certainty in its prediction")
        st.caption("Streaming Simulation = Real-time row-by-row intrusion demo")
        st.markdown("""
        - Trained on the **NSL-KDD dataset**: A widely used benchmark dataset for intrusion detection.
        - Features are one-hot encoded and standardized: Data preprocessing steps for model readiness.
        - Model: **Random Forest Classifier** or **Logistic Regression**: You can choose which model to use.
        - Predicts if traffic is **Attack** or **Normal**: Binary classification task.
        - Includes confidence score and live streaming preview: Provides insight into model certainty and real-time demonstration.
        - **Simulated Attack Map:** The latitude and longitude coordinates for the attack map are randomly generated for demonstration purposes and do not reflect actual geographical locations.
        """)

    st.markdown("""
    ---
    <p style='text-align: center; font-size: 0.9em;'>🔒 Powered by Machine Learning | Streamlit App by <b>Prasamita Bangal.</b></p>
    """, unsafe_allow_html=True)
else:
    st.info("👆 Please upload your `preprocessed_test_data.csv` file or click 'Try with Sample Data' to see predictions. Ensure 'ids_model.pkl', 'logistic_model.pkl', and 'model_features.pkl' are in the same directory.")

