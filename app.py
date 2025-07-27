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
from reportlab.lib.units import inch # Import inch for image sizing

st.set_page_config(page_title="AI IDS", layout="centered")

# --- Start of new code for cyber grid background and overall styling ---
cyber_grid_css = """
<style>
/* Base body styling for the dark theme */
body {
    background-color: #1a1a2e; /* Deep dark blue/purple for the overall background */
    overflow-x: hidden; /* Hide horizontal scrollbar */
    font-family: 'Inter', sans-serif; /* A clean, modern font */
    color: #e0e0e0; /* Default text color */
}

/* Animated Cyber Grid Background */
body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1; /* Place behind content */
    background-image:
        linear-gradient(0deg, transparent 99%, rgba(0, 255, 255, 0.08) 100%), /* Subtle horizontal lines (cyan/teal) */
        linear-gradient(90deg, transparent 99%, rgba(0, 255, 255, 0.08) 100%); /* Subtle vertical lines */
    background-size: 60px 60px; /* Adjust grid cell size for more subtle effect */
    opacity: 0.2; /* Very subtle grid */
    animation: grid-movement 90s linear infinite; /* Slower, continuous movement */
    pointer-events: none; /* Allow interaction with elements behind it */
}

@keyframes grid-movement {
    0% {
        background-position: 0 0;
    }
    100% {
        background-position: 60px 60px; /* Moves by one grid cell */
    }
}

/* Streamlit main app container */
.stApp {
    background-color: transparent; /* Let the body background show through */
}

/* Main content block container */
.main .block-container {
    background-color: rgba(30, 30, 46, 0.9); /* Slightly lighter, semi-transparent dark background for content */
    border-radius: 12px; /* More rounded corners */
    padding: 30px; /* More padding */
    box-shadow: 0 4px 15px rgba(0, 255, 255, 0.1); /* Subtle glow effect */
    margin-top: 20px; /* Space from the top header */
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #00e0ff; /* Bright cyan for headings */
    font-family: 'Inter', sans-serif;
    text-shadow: 0 0 5px rgba(0, 255, 255, 0.3); /* Subtle text glow */
}

/* Paragraphs and general text */
p, li, .stMarkdown {
    color: #c0c0c0; /* Light gray for body text */
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
}

/* Buttons */
.stButton>button {
    background-color: #007bff; /* A standard blue */
    background-image: linear-gradient(45deg, #007bff, #00c0e0); /* Gradient for buttons */
    color: white;
    border: none;
    border-radius: 8px; /* Rounded buttons */
    padding: 12px 25px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s ease; /* Smooth transitions for hover */
    box-shadow: 0 4px 10px rgba(0, 123, 255, 0.3); /* Button shadow */
    cursor: pointer;
}
.stButton>button:hover {
    background-image: linear-gradient(45deg, #0056b3, #0099b3); /* Darker gradient on hover */
    box-shadow: 0 6px 15px rgba(0, 123, 255, 0.5); /* Enhanced shadow on hover */
    transform: translateY(-2px); /* Slight lift effect */
}

/* Selectboxes and Text Inputs */
.stSelectbox>div>div, .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stFileUploader>section {
    background-color: #2a2a40; /* Darker background for input fields */
    color: #e0e0e0; /* Text color inside inputs */
    border: 1px solid #00aaff; /* Border with accent color */
    border-radius: 8px;
    padding: 8px 12px;
}
.stSelectbox .css-1dbjc4n-base-Input { /* Targeting the input part of selectbox */
    color: #e0e0e0;
}
.stSelectbox .css-1dbjc4n-base-Input::placeholder {
    color: #888;
}

/* Dataframes */
.stDataFrame {
    color: #e0e0e0;
    background-color: #2a2a40;
    border-radius: 8px;
    overflow: hidden; /* Ensures rounded corners apply to content */
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}
.stDataFrame table {
    background-color: #2a2a40;
    width: 100%; /* Ensure table takes full width */
}
.stDataFrame th {
    background-color: #3a3a50; /* Header background */
    color: #00e0ff; /* Header text with accent */
    font-weight: bold;
    padding: 10px;
    text-align: left;
}
.stDataFrame td {
    padding: 10px;
    border-bottom: 1px solid #3a3a50; /* Subtle row separator */
}
.stDataFrame tr:nth-child(even) {
    background-color: #2e2e42; /* Zebra striping for readability */
}

/* Sidebar styling */
.stSidebar {
    background-color: #1a1a2e; /* Match main body background */
    color: #e0e0e0;
    border-right: 2px solid #00aaff; /* Accent line on the right */
    box-shadow: 2px 0 10px rgba(0, 255, 255, 0.1); /* Subtle shadow */
}
.stSidebar .stRadio div[role="radiogroup"] label {
    color: #e0e0e0; /* Radio button text color */
}
.stSidebar h1, .stSidebar h2, .stSidebar h3 {
    color: #00e0ff; /* Sidebar headings */
}

/* Expander styling */
.stExpander {
    background-color: #2a2a40;
    border-radius: 8px;
    padding: 15px;
    margin-top: 20px;
    border: 1px solid #00aaff;
}
.stExpander > div > div > p {
    color: #e0e0e0; /* Expander header text */
    font-weight: bold;
}
.stExpander .streamlit-expanderContent {
    color: #c0c0c0; /* Expander content text */
}

/* Metric styling */
.stMetric {
    background-color: #2a2a40;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(0, 255, 255, 0.2);
}
.stMetric > div > div > div > div {
    color: #00e0ff; /* Metric value color */
    font-size: 2.5rem; /* Larger font for value */
    font-weight: bold;
}
.stMetric > div > div > div > label {
    color: #c0c0c0; /* Metric label color */
    font-size: 1rem;
    font-weight: normal;
}

/* Horizontal Rule */
hr {
    border-top: 2px solid #00aaff; /* Thicker, accent-colored HR */
    margin: 2em 0;
}

/* Caption styling */
.stCaption {
    color: #a0a0a0;
    font-size: 0.85em;
    margin-top: 5px;
}

/* Specific styling for the initial header div */
div[data-testid="stMarkdownContainer"] div[style*="padding: 10px;"] {
    background-color: rgba(30, 30, 46, 0.9); /* Ensure this specific div matches the main container */
    border-radius: 12px;
    padding: 10px; /* Keep original padding */
    box-shadow: 0 4px 15px rgba(0, 255, 255, 0.1);
    margin-bottom: 20px; /* Space below the header */
}

</style>
"""

st.markdown(cyber_grid_css, unsafe_allow_html=True)
# --- End of new code for cyber grid background and overall styling ---


# Theme toggle (existing logic, but now the default dark theme is enhanced by CSS)
if "theme" not in st.session_state:
    st.session_state.theme = "Dark" # Set default to Dark theme

chosen_theme = st.sidebar.radio(
    "Choose Theme", ["Light", "Dark"],
    index=0 if st.session_state.theme == "Light" else 1 # This will still work for the toggle, but the CSS overrides the background.
)
st.session_state.theme = chosen_theme

# The light_style/dark_style variables will primarily affect the text color within the initial markdown div
# The overall background is now handled by the injected CSS.
light_style = "background-color: transparent; color: black;" # Transparent to let grid show
dark_style = "background-color: transparent; color: white;" # Transparent to let grid show
style = light_style if st.session_state.theme == "Light" else dark_style


st.image("logo.png", width=100)

st.markdown(f"""
<div style='{style} padding: 10px;'>
<h1 style='text-align: center;'> AI-Powered Intrusion Detection System 🔐</h1>
<h4 style='text-align: center;'>Detect cyber attacks in real-time using Machine Learning</h4>
<p style='text-align: center; font-size: 14px;'>Built by <b>Prasamita Bangal.</b> | Mahindra University</p></div>
<hr style='border-top: 2px solid #bbb;'>
""", unsafe_allow_html=True)

# Model selector
model_choice = st.sidebar.selectbox(
    "Choose a Model", ["Random Forest", "Logistic Regression"]
)

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
    ## Navigation
    - 1. Data Preview
    - 2. Summary Metrics
    - 3. Prediction Breakdown
    - 4. Model Accuracy & Confusion Matrix
    - 5. Top 10 Feature Importances
    - 6. Live Streaming Simulation
    - 7. Simulated Attack Map
    - 8. Full Predictions (Top 25)
    """, unsafe_allow_html=True)
    st.markdown("""
    This is a lightweight, AI-powered intrusion detection dashboard built with:
    - Random Forest or Logistic Regression
    - NSL-KDD dataset
    - Real-time visualization
    Try uploading a CSV or use sample data to test the IDS engine.
    """)

    st.markdown("---")
    st.subheader("Feedback Form")
    feedback_name = st.text_input("Your Name")
    feedback_text = st.text_area("Your Feedback")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
        with open("feedback_log.txt", "a") as f:
            f.write(f"{datetime.now()} - {feedback_name}: {feedback_text}\n")

# Load data
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

    st.subheader("1. Data Preview")
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
    st.subheader("2. Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Attacks Detected", total_attacks)
    with col3:
        st.metric("Normal Traffic", total_normal)
    with col4:
        st.metric("Attack %", f"{(total_attacks / len(data)) * 100:.2f}%")

    # Removed the redundant dataframe for summary metrics as st.metric is better
    # st.dataframe(pd.DataFrame({
    #     "Category": ["Total Records", "Attacks Detected", "Normal Traffic", "Attack %"],
    #     "Value": [len(data), total_attacks, total_normal, f"{(total_attacks / len(data)) * 100:.2f}%"]
    # }))

    st.markdown("---")
    st.subheader("3. Prediction Breakdown")
    pred_counts = data["Prediction"].value_counts()
    colors = ["#2ecc71" if label == "Normal" else "#e74c3c" for label in pred_counts.index]
    fig, ax = plt.subplots()
    ax.bar(pred_counts.index, pred_counts.values, color=colors)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    ax.set_title("Attack vs Normal")
    ax.grid(True, linestyle='--', alpha=0.5)
    # Set plot background and text colors for dark theme
    fig.patch.set_facecolor('#2a2a40') # Plot background
    ax.set_facecolor('#2a2a40') # Axes background
    ax.tick_params(axis='x', colors='#e0e0e0') # X-axis tick labels
    ax.tick_params(axis='y', colors='#e0e0e0') # Y-axis tick labels
    ax.xaxis.label.set_color('#00e0ff') # X-axis label color
    ax.yaxis.label.set_color('#00e0ff') # Y-axis label color
    ax.title.set_color('#00e0ff') # Title color
    ax.spines['bottom'].set_color('#00aaff') # Axis lines
    ax.spines['left'].set_color('#00aaff')
    ax.spines['top'].set_color('#00aaff')
    ax.spines['right'].set_color('#00aaff')

    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='#e0e0e0') # Annotation color
    st.pyplot(fig)
    st.caption("🟢 Green = Normal   🔴 Red = Attack")

    st.markdown("---")
    st.subheader("4. Model Accuracy & Confusion Matrix")
    true_labels = [1 if lbl == "Normal" else 0 for lbl in labels]
    cm = confusion_matrix(true_labels, predictions)
    acc = accuracy_score(true_labels, predictions)

    st.write(f"**Accuracy Score:** <span style='color:#00e0ff; font-size:1.2em;'>{acc*100:.2f}%</span>", unsafe_allow_html=True)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Attack", "Normal"],
                yticklabels=["Attack", "Normal"], ax=ax_cm,
                annot_kws={"color": "black"}) # Ensure annotation text is visible
    ax_cm.set_xlabel("Predicted", color='#00e0ff')
    ax_cm.set_ylabel("Actual", color='#00e0ff')
    ax_cm.tick_params(axis='x', colors='#e0e0e0')
    ax_cm.tick_params(axis='y', colors='#e0e0e0')
    fig_cm.patch.set_facecolor('#2a2a40') # Plot background
    ax_cm.set_facecolor('#2a2a40') # Axes background
    st.pyplot(fig_cm)

    st.markdown("---")
    st.subheader("5. Top 10 Feature Importances")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_series = pd.Series(importances, index=feature_list).sort_values(ascending=False).head(10)
        # Streamlit's bar_chart uses its own styling, which usually adapts well.
        st.bar_chart(feat_series)
    else:
        st.info("Feature importances not available for this model.")

    st.markdown("---")
    st.subheader("6. 📺 Live Streaming Simulation")
    if st.button("▶️ Start Stream Simulation"):
        import time
        live_placeholder = st.empty()
        for i in range(min(25, len(data))):
            live_row = data.iloc[[i]][feature_list]
            pred = model.predict(live_row)[0]
            label = "Normal" if pred == 1 else "Attack"
            conf = f"{max(model.predict_proba(live_row)[0]) * 100:.2f}%"
            color_code = "green" if label == "Normal" else "red"
            live_placeholder.markdown(f"**Row {i+1}:** `<span style='color:{color_code}; font-weight:bold;'>{label}</span>` (Confidence: {conf})", unsafe_allow_html=True)
            time.sleep(0.6)

    st.markdown("---")
    st.subheader("7. Simulated Attack Map")
    try:
        attack_data = data[data['Prediction'] == 'Attack'].copy()
        if not attack_data.empty:
            attack_data['lat'] = np.random.uniform(8.0, 37.0, len(attack_data))
            attack_data['lon'] = np.random.uniform(68.0, 97.0, len(attack_data))
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/dark-v9', # Force dark map style for consistency
                initial_view_state=pdk.ViewState(latitude=22.0, longitude=78.0, zoom=3.5),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=attack_data,
                        get_position='[lon, lat]',
                        get_color='[255, 0, 0, 180]', # Brighter red for attacks
                        get_radius=50000, # Slightly larger radius
                    )
                ]
            ))
        else:
            st.info("No attacks detected to map.")
    except Exception as e:
        st.error(f"Could not generate attack map: {str(e)}")

    st.markdown("---")
    st.subheader("8. 📄 Full Predictions (Top 25)")
    display_data = data.copy()
    display_data.insert(0, "🔒 Threat", data["Prediction"])
    st.dataframe(display_data.head(25))

    st.download_button(
        label="Download Predictions CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="IDS_predictions.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("🔖 Generate PDF Report")
    if st.button("📤 Generate PDF Summary"):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        # Custom styles for PDF to match dark theme aesthetics (simplified)
        styles.add(ParagraphStyle(name='TitleDark', fontSize=24, leading=28, alignment=TA_CENTER,
                                  fontName='Helvetica-Bold', textColor=colors.HexColor('#00e0ff')))
        styles.add(ParagraphStyle(name='NormalDark', fontSize=12, leading=14,
                                  fontName='Helvetica', textColor=colors.HexColor('#c0c0c0')))
        styles.add(ParagraphStyle(name='Heading1Dark', fontSize=18, leading=22,
                                  fontName='Helvetica-Bold', textColor=colors.HexColor('#00e0ff')))

        elements = [
            Image("logo.png", width=1.5*inch, height=0.9*inch), # Use inch for sizing
            Spacer(1, 12),
            Paragraph("AI-Powered Intrusion Detection Report", styles['TitleDark']),
            Spacer(1, 24),
            Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['NormalDark']),
            Spacer(1, 12),
            Paragraph(f"Model Used: {model_choice}", styles['NormalDark']),
            Paragraph(f"Total Records: {len(data)}", styles['NormalDark']),
            Paragraph(f"Attacks Detected: {total_attacks}", styles['NormalDark']),
            Paragraph(f"Normal Traffic: {total_normal}", styles['NormalDark']),
            Paragraph(f"Attack %: {(total_attacks / len(data)) * 100:.2f}%", styles['NormalDark']),
            Spacer(1, 24),
            Paragraph("This report summarizes the intrusion detection results for the uploaded network traffic logs using AI-based classifiers.", styles['NormalDark']),
            Spacer(1, 36),
            Paragraph("Generated by Prasamita Bangal.", styles['NormalDark']),
            Paragraph("Mahindra University", styles['NormalDark']),
        ]
        doc.build(elements)
        st.download_button(
            label="Download PDF Report",
            data=buffer.getvalue(),
            file_name="IDS_Report.pdf",
            mime="application/pdf"
        )

    st.markdown("---")
    with st.expander("How This Works"):
        st.caption("🟢 Green = Normal Traffic  🔴 Red = Attack Traffic")
        st.caption("Confidence = Model's certainty in its prediction")
        st.caption("Streaming Simulation = Real-time row-by-row intrusion demo")
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

