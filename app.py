import streamlit as st
import joblib
import numpy as np

# 1. Modeli ve scaler'ı yükle
# Not: .pkl dosyalarınla aynı klasörde olduğundan emin ol
model = joblib.load('vucut_yagi_modeli.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Vücut Yağı Tahmincisi 📊")
st.write("Verilerini gir, yapay zeka vücut yağ oranını hesaplasın.")

# 2. Giriş alanları (Sıra: Age, Weight, Neck, Abdomen, Ankle, Biceps, Forearm, Wrist)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Yaş", min_value=15, max_value=90, value=22)
    weight = st.number_input("Kilo (kg)", min_value=40.0, max_value=150.0, value=75.0)
    neck = st.number_input("Boyun (cm)", min_value=20.0, max_value=60.0, value=38.0)
    abdomen = st.number_input("Karın (cm)", min_value=50.0, max_value=150.0, value=85.0)

with col2:
    ankle = st.number_input("Ayak Bileği (cm)", min_value=10.0, max_value=40.0, value=23.0)
    biceps = st.number_input("Pazı (cm)", min_value=15.0, max_value=60.0, value=33.0)
    forearm = st.number_input("Ön Kol (cm)", min_value=10.0, max_value=50.0, value=29.0)
    wrist = st.number_input("Bilek (cm)", min_value=10.0, max_value=30.0, value=18.0)

# 3. Tahmin İşlemi
if st.button("Tahmin Et"):
    features = np.array([[age, weight, neck, abdomen, ankle, biceps, forearm, wrist]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    st.success(f"Tahmin edilen vücut yağı oranı: %{prediction[0]:.2f}")