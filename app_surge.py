import streamlit as st
import pandas as pd
import joblib
import json

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Surge Pricing", layout="centered")
st.title("🚖 Prediksi Surge Pricing Taksi Online")
st.write("Masukkan informasi perjalanan untuk memprediksi tingkat surge pricing.")

# Load model dan feature order
model = joblib.load("xgb_surge_model.joblib")
with open("feature_order.json") as f:
    feature_order = json.load(f)

# Mapping untuk fitur kategorikal
type_of_cab_map = {
    "Mini": 0,
    "Sedan": 1,
    "SUV": 2,
    "Luxury": 3
}

destination_map = {
    "Airport": 0,
    "City Center": 1,
    "Hotel": 2,
    "Mall": 3,
    "Other": 4
}

gender_map = {
    "Male": 1,
    "Female": 0
}

# Input pengguna
type_of_cab_label = st.selectbox("Type of Cab", list(type_of_cab_map.keys()))
trip_distance = st.slider("Trip Distance (km)", 0.0, 50.0, 5.0)
customer_rating = st.slider("Customer Rating", 0.0, 5.0, 4.0)
life_style_index = st.slider("Life Style Index", 0.0, 1.0, 0.5)
confidence_life_style = st.slider("Confidence Life Style Index", 0.0, 1.0, 0.5)
destination_label = st.selectbox("Destination Type", list(destination_map.keys()))
cancellation_1m = st.slider("Cancellation Last 1 Month", 0, 5, 1)
gender_label = st.radio("Gender", list(gender_map.keys()))

# Input dummy untuk fitur lain
input_dict = {
    'Trip_Distance': trip_distance,
    'Type_of_Cab': type_of_cab_map[type_of_cab_label],
    'Customer_Since_Months': 12,  # asumsi rata-rata
    'Life_Style_Index': life_style_index,
    'Confidence_Life_Style_Index': confidence_life_style,
    'Destination_Type': destination_map[destination_label],
    'Customer_Rating': customer_rating,
    'Cancellation_Last_1Month': cancellation_1m,
    'Var2': 0,
    'Var3': 0,
    'Gender': gender_map[gender_label]
}

# Susun input menjadi DataFrame sesuai urutan fitur
data = pd.DataFrame([[input_dict[feat] for feat in feature_order]], columns=feature_order)

# Prediksi
if st.button("🔍 Prediksi Surge Pricing"):
    pred = model.predict(data)[0]
    label_map = {0: "Rendah (0)", 1: "Sedang (1)", 2: "Tinggi (2)"}
    st.success(f"Hasil Prediksi: **{label_map[pred]}**")
