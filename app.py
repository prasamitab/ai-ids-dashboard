import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AI IDS", layout="centered")
st.title(" AI-Powered Intrusion Detection System")
st.markdown("Upload preprocessed network log data and detect cyber attacks using AI.")

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
    labels = ["Normal" if i == 1 else "Attack" for i in predictions]
    data['Prediction'] = labels

    # Prediction Summary
    st.subheader("📊 Prediction Summary")
    summary = data['Prediction'].value_counts()
    st.write(summary)
    st.bar_chart(summary)

    # Full result preview
    st.subheader("📄 Full Predictions")
    st.dataframe(data.head(25))

    # Download option
    st.download_button(
        label="📥 Download Results",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name="IDS_predictions.csv",
        mime="text/csv"
    )
else:
    st.info("👆 Please upload your `preprocessed_test_data.csv` file to see predictions.")
