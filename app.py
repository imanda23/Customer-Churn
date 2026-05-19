import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# 1. Load Resources (Encoders, Scaler, Model)
@st.cache_resource
def load_resources():
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('best_model_churn.pkl', 'rb') as f:
        model = pickle.load(f)
    return encoders, scaler, model

try:
    loaded_encoders, loaded_scaler, loaded_model = load_resources()
except FileNotFoundError:
    st.error("File pendukung (model/scaler/encoder) tidak ditemukan. Pastikan cell sebelumnya telah dijalankan.")
    st.stop()

# 2. UI App Layout
st.title("📱 Customer Churn Prediction App")
st.markdown("Aplikasi ini memprediksi apakah seorang pelanggan akan berhenti berlangganan (**Churn**) atau tidak.")

st.sidebar.header("Input Data Pelanggan")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", options=['Female', 'Male'])
    tenure = st.sidebar.slider("Tenure (Bulan)", 1, 60, 24)
    usage_freq = st.sidebar.slider("Usage Frequency (Kali/Bulan)", 1, 30, 15)
    support_calls = st.sidebar.slider("Support Calls (Kali)", 0, 10, 2)
    payment_delay = st.sidebar.slider("Payment Delay (Hari)", 0, 30, 5)
    sub_type = st.sidebar.selectbox("Subscription Type", options=['Basic', 'Standard', 'Premium'])
    contract = st.sidebar.selectbox("Contract Length", options=['Monthly', 'Quarterly', 'Annual'])

    data = {
        'Gender': gender,
        'Tenure': tenure,
        'Usage Frequency': usage_freq,
        'Support Calls': support_calls,
        'Payment Delay': payment_delay,
        'Subscription Type': sub_type,
        'Contract Length': contract
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 3. Pre-processing Input Data
st.subheader("Data Input Pelanggan")
st.write(input_df)

# Clone data for processing
processed_df = input_df.copy()

# Apply Label Encoding
for col, le in loaded_encoders.items():
    if col in processed_df.columns:
        processed_df[col] = le.transform(processed_df[col])

# Define the exact order of columns used during scaling
features_order = ['Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
                  'Payment Delay', 'Subscription Type', 'Contract Length']

# Apply Scaling
scaled_features = loaded_scaler.transform(processed_df[features_order])

# 4. Prediction
if st.button("Prediksi Sekarang"):
    prediction = loaded_model.predict(scaled_features)
    prediction_proba = loaded_model.predict_proba(scaled_features)[:, 1]

    st.divider()
    st.subheader("Hasil Analisis:")

    if prediction[0] == 1:
        st.error(f"🚩 Pelanggan Berpotensi **CHURN** (Probabilitas: {prediction_proba[0]:.2%})")
        st.info("Rekomendasi: Berikan penawaran khusus atau diskon retensi.")
    else:
        st.success(f"✅ Pelanggan **TIDAK CHURN** (Probabilitas: {1 - prediction_proba[0]:.2%})")
        st.info("Rekomendasi: Jaga kepuasan dengan program loyalty.")

st.markdown("---")
st.caption("Dikembangkan menggunakan Gradient Boosting + SMOTE Pipeline.")
