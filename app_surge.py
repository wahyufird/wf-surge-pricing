import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Surge Pricing", layout="centered")

st.title("🚖 Prediksi Surge Pricing Taksi Online")
st.write("Masukkan informasi perjalanan untuk memprediksi tingkat surge pricing.")

# Load model
model = joblib.load("xgb_surge_model.joblib")

# Load urutan fitur saat training
with open("feature_order.json") as f:
    feature_order = json.load(f)

# Mapping label asli
cab_mapping = {"Basic": 0, "Economy": 1, "Premium": 2, "Luxury": 3}
destination_mapping = {"Airport": 0, "City Center": 1, "Hotel": 2, "Mall": 3, "Other": 4}

# Input pengguna
type_of_cab_label = st.selectbox("🚕 Tipe Taksi", list(cab_mapping.keys()))
trip_distance = st.slider("📏 Jarak Perjalanan (km)", 0.0, 50.0, 5.0)
customer_rating = st.slider("⭐ Rating Pelanggan", 0.0, 5.0, 4.0)
life_style_index = st.slider("💼 Life Style Index", 0.0, 1.0, 0.5)
confidence_life_style = st.slider("📊 Confidence Life Style Index", 0.0, 1.0, 0.5)
destination_label = st.selectbox("📍 Tujuan Perjalanan", list(destination_mapping.keys()))
cancellation_1m = st.slider("❌ Pembatalan dalam 1 Bulan", 0, 5, 1)

# Input dummy untuk fitur lain agar sesuai model
input_dict = {
    'Trip_Distance': trip_distance,
    'Type_of_Cab': cab_mapping[type_of_cab_label],
    'Customer_Since_Months': 12,
    'Life_Style_Index': life_style_index,
    'Confidence_Life_Style_Index': confidence_life_style,
    'Destination_Type': destination_mapping[destination_label],
    'Customer_Rating': customer_rating,
    'Cancellation_Last_1Month': cancellation_1m,
    'Var2': 0,
    'Var3': 0,
    'Gender': 1
}

# Susun input sesuai urutan fitur saat training
data = pd.DataFrame([[input_dict[feat] for feat in feature_order]], columns=feature_order)

# Prediksi dan tampilkan hasil
if st.button("🔍 Prediksi Surge Pricing"):
    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0]

    label_map = {0: "Rendah (0)", 1: "Sedang (1)", 2: "Tinggi (2)"}
    st.success(f"Hasil Prediksi: **{label_map[pred]}**")

    # Visualisasi probabilitas prediksi
    st.write("### 🔢 Probabilitas Tiap Kelas")
    fig, ax = plt.subplots()
    ax.bar(label_map.values(), proba, color=["green", "orange", "red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilitas")
    ax.set_title("Confidence Score")
    st.pyplot(fig)
