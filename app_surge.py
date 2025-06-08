import streamlit as st
import pandas as pd
import joblib
import json

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Surge Pricing", layout="centered")

st.title("🚖 Prediksi Surge Pricing Taksi Online")
st.write("Masukkan informasi perjalanan untuk memprediksi tingkat surge pricing.")

# Load model
model = joblib.load("xgb_surge_model.joblib")

# Load urutan fitur saat training
with open("feature_order.json") as f:
    feature_order = json.load(f)

# Input pengguna
type_of_cab = st.selectbox("Type of Cab", [0, 1, 2, 3])
trip_distance = st.slider("Trip Distance", 0.0, 50.0, 5.0)
customer_rating = st.slider("Customer Rating", 0.0, 5.0, 4.0)
life_style_index = st.slider("Life Style Index", 0.0, 1.0, 0.5)
confidence_life_style = st.slider("Confidence Life Style Index", 0.0, 1.0, 0.5)
destination_type = st.selectbox("Destination Type", [0, 1])
cancellation_1m = st.slider("Cancellation Last 1 Month", 0, 5, 1)

# Input dummy untuk fitur lain agar sesuai dengan fitur model
input_dict = {
    'Trip_Distance': trip_distance,
    'Type_of_Cab': type_of_cab,
    'Customer_Since_Months': 12,         # nilai default (bisa diganti rata-rata)
    'Life_Style_Index': life_style_index,
    'Confidence_Life_Style_Index': confidence_life_style,
    'Destination_Type': destination_type,
    'Customer_Rating': customer_rating,
    'Cancellation_Last_1Month': cancellation_1m,
    'Var2': 0,                            # default atau bisa berdasarkan nilai umum
    'Var3': 0,
    'Gender': 1                           # default (1 = male atau female sesuai model)
}

# Susun input menjadi DataFrame sesuai urutan fitur saat training
data = pd.DataFrame([[input_dict[feat] for feat in feature_order]], columns=feature_order)

# Prediksi
if st.button("🔍 Prediksi Surge Pricing"):
    pred = model.predict(data)[0]
    label_map = {0: "Rendah (0)", 1: "Sedang (1)", 2: "Tinggi (2)"}
    st.success(f"Hasil Prediksi: {label_map[pred]}")
