import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="AI Intrusion Detection System", layout="centered")

# Title
st.title("🔐 AI-Powered Intrusion Detection System")
st.markdown("This dashboard uses a machine learning model trained on the NSL-KDD dataset to detect whether incoming network traffic is **Normal** or **Attack**.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("ids_model.pkl")

model = load_model()

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your preprocessed network data (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df.head())

    # Predict
    predictions = model.predict(df)
    labels = ["Normal" if p == 1 else "Attack" for p in predictions]
    df['Prediction'] = labels

    # Show summary
    st.subheader("📊 Prediction Summary")
    st.write(df['Prediction'].value_counts())
    st.bar_chart(df['Prediction'].value_counts())

    # Show full results
    st.subheader("📄 Full Predictions")
    st.dataframe(df.head(25))

    # Download results
    st.download_button(
        label="📥 Download Result CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="IDS_Predictions.csv",
        mime="text/csv"
    )
else:
    st.info("👆 Please upload your `preprocessed_test_data.csv` file to see predictions.")
