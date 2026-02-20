import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AQI Monitoring System", layout="wide")

# ---------------- IMPORT CUSTOM MODULE ----------------
sys.path.append(os.path.abspath("."))
from src.aqi_utils import get_aqi_category, get_health_advisory

# ---------------- LOAD MODEL & DATA ----------------
model = joblib.load("Model/aqi_model.pkl")
df = pd.read_csv("Data/cleaned_air_quality.csv")

# ---------------- TITLE ----------------
st.title("🌍 AQI Monitoring System Dashboard")

st.sidebar.header("About")
st.sidebar.write(
    "This application predicts Air Quality Index (AQI) using a "
    "Machine Learning model and provides city-wise analysis."
)

# ======================================================
# SECTION 1 — CITY & YEAR ANALYSIS
# ======================================================

st.header("📍 City & Year Analysis")

col1, col2 = st.columns(2)

with col1:
    cities = sorted(df["City"].unique())
    selected_city = st.selectbox("Select City", cities)

with col2:
    years = sorted(df["Year"].unique())
    selected_year = st.selectbox("Select Year", years)

# Filter data
filtered_data = df[(df["City"] == selected_city) & (df["Year"] == selected_year)]

if not filtered_data.empty:
    avg_aqi = round(filtered_data["AQI"].mean(), 2)
    st.success(f"Average AQI in {selected_city} ({selected_year}) = {avg_aqi}")
else:
    st.warning("No data available for selected city and year.")

# Yearly Trend Graph
st.subheader(f"📈 AQI Trend for {selected_city}")

city_data = df[df["City"] == selected_city]
yearly_avg = city_data.groupby("Year")["AQI"].mean()

st.line_chart(yearly_avg)

# ======================================================
# SECTION 2 — AQI PREDICTION
# ======================================================

st.header("🔮 Predict AQI Using Pollutant Values")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5", min_value=0.0)
    pm10 = st.number_input("PM10", min_value=0.0)
    no2 = st.number_input("NO2", min_value=0.0)
    nh3 = st.number_input("NH3", min_value=0.0)

with col2:
    co = st.number_input("CO", min_value=0.0)
    so2 = st.number_input("SO2", min_value=0.0)
    o3 = st.number_input("O3", min_value=0.0)

if st.button("Predict AQI"):

    input_data = [pm25, pm10, no2, nh3, co, so2, o3]

    prediction = model.predict([input_data])[0]
    aqi_value = round(prediction, 2)

    category = get_aqi_category(aqi_value)
    advisory = get_health_advisory(category)

    # Color Logic
    if aqi_value <= 50:
        color = "green"
    elif aqi_value <= 100:
        color = "#7CFC00"
    elif aqi_value <= 200:
        color = "orange"
    elif aqi_value <= 300:
        color = "red"
    elif aqi_value <= 400:
        color = "darkred"
    else:
        color = "purple"

    st.markdown("## 📊 Prediction Result")

    st.markdown(f"""
    <div style="background-color:#111; padding:20px; border-radius:10px;">
        <h2 style="color:{color};">AQI: {aqi_value}</h2>
        <h4>Category: {category}</h4>
        <p>{advisory}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- POLLUTANT BAR CHART ----------------
    st.subheader("📊 Pollutant Levels Entered")

    pollutants = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "NH3": nh3,
        "CO": co,
        "SO2": so2,
        "O3": o3
    }

    fig, ax = plt.subplots()
    ax.bar(pollutants.keys(), pollutants.values())
    ax.set_ylabel("Value")
    ax.set_title("Pollutant Levels")

    st.pyplot(fig)

# ======================================================
# SECTION 3 — AQI SCALE REFERENCE
# ======================================================

st.header("🌈 AQI Scale Reference")

st.markdown("""
- 🟢 0–50 → Good  
- 🟡 51–100 → Satisfactory  
- 🟠 101–200 → Moderate  
- 🔴 201–300 → Poor  
- 🟣 301–400 → Very Poor  
- ⚫ 401–500 → Severe  
""")