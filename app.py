# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load models and scaler
lr_model = joblib.load("models/lr_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
gb_model = joblib.load("models/gb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load dataset to get feature names
data = load_breast_cancer()
feature_names = data.feature_names

st.title("🧬 Breast Cancer Prediction Web App (CSV Upload)")
st.markdown("Upload a CSV file with **exactly 30 features** to predict whether each case is **Malignant or Benign**.")

# Model selection
model_choice = st.selectbox("🔍 Choose Classifier", ("Logistic Regression", "Random Forest", "Gradient Boosting"))

# File upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if list(df.columns) != list(feature_names):
        st.error("❌ CSV must have exactly 30 columns:\n" + ", ".join(feature_names))
    else:
        st.success("✅ File successfully loaded!")

        input_scaled = scaler.transform(df.values)

        # Predict
        model = {"Logistic Regression": lr_model, "Random Forest": rf_model, "Gradient Boosting": gb_model}[model_choice]
        predictions = model.predict(input_scaled)
        prediction_labels = np.where(predictions == 1, "Benign", "Malignant")
        df["Prediction"] = prediction_labels

        if df.shape[0] == 1:
            result = prediction_labels[0]
            color = "🟢" if result == "Benign" else "🔴"
            st.subheader("🧾 Diagnosis Result")
            st.success(f"{color} The tumor is predicted to be: **{result}** means it is not cancerous." if result == "Benign" else f"{color} The tumor is predicted to be: **{result}** means it is cancerous.")
        else:
            st.subheader("📋 Prediction Table")
            st.dataframe(df)

            # Pie chart
            st.subheader("📊 Prediction Summary Chart")
            benign_count = sum(predictions == 1)
            malignant_count = sum(predictions == 0)
            labels = ['Benign', 'Malignant']
            sizes = [benign_count, malignant_count]
            colors = ['#28a745', '#dc3545']
            explode = (0.1, 0)
            fig, ax = plt.subplots()
            ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=140)
            ax.axis('equal')
            st.pyplot(fig)

            # CSV Download
            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions as CSV", csv_download, "predictions.csv", "text/csv")

