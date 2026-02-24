import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AQI Monitoring System", layout="wide")

# ---------------- IMPORT UTILS ----------------
sys.path.append(os.path.abspath("."))
from src.aqi_utils import get_aqi_category, get_health_advisory

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Data/cleaned_air_quality.csv")

# ---------------- LOAD MODEL ----------------
model = joblib.load("Model/aqi_model.pkl")

# ---------------- TITLE ----------------
st.title("🌍 Air Quality Monitoring & Prediction System")

st.sidebar.header("About")
st.sidebar.write(
    "This system predicts AQI automatically using historical pollutant data "
    "and a trained Machine Learning model."
)

# =====================================================
# SECTION 1 — CITY & YEAR SELECTION
# =====================================================
st.header("📍 Select City and Year")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", sorted(df["City"].unique()))

with col2:
    year = st.selectbox("Year", sorted(df["Year"].unique()))

filtered = df[(df["City"] == city) & (df["Year"] == year)]

# =====================================================
# SECTION 2 — AQI ANALYSIS
# =====================================================
st.header("📊 AQI Analysis")

if filtered.empty:
    st.warning("No data available for this city and year.")
    st.stop()

avg_aqi = round(filtered["AQI"].mean(), 2)
st.success(f"Average AQI in {city} ({year}) = {avg_aqi}")

# Year-wise trend
st.subheader(f"📈 AQI Trend for {city}")
trend = df[df["City"] == city].groupby("Year")["AQI"].mean()
st.line_chart(trend)

# =====================================================
# SECTION 3 — AUTOMATIC AQI PREDICTION
# =====================================================
st.header("🔮 AQI Prediction (Automatic)")

pollutant_cols = ["PM2.5", "PM10", "NO2", "NH3", "CO", "SO2", "O3"]
pollutant_values = filtered[pollutant_cols].mean()

if st.button("Predict AQI"):
    prediction = model.predict([pollutant_values.values])[0]
    aqi_value = round(prediction, 2)

    category = get_aqi_category(aqi_value)
    advisory = get_health_advisory(category)

    # Color logic
    if aqi_value <= 50:
        color = "green"
    elif aqi_value <= 100:
        color = "#9ACD32"
    elif aqi_value <= 200:
        color = "orange"
    elif aqi_value <= 300:
        color = "red"
    else:
        color = "purple"

    st.markdown("## 📌 Prediction Result")
    st.markdown(
        f"""
        <div style="background-color:#111;padding:20px;border-radius:10px;">
            <h2 style="color:{color};">AQI: {aqi_value}</h2>
            <h4>Category: {category}</h4>
            <p>{advisory}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Pollutant bar chart
    st.subheader("📊 Average Pollutant Levels")
    fig, ax = plt.subplots()
    ax.bar(pollutant_cols, pollutant_values)
    ax.set_ylabel("Concentration")
    st.pyplot(fig)

# =====================================================
# SECTION 4 — AQI SCALE
# =====================================================
st.header("🌈 AQI Scale Reference")
st.markdown("""
- 🟢 **0–50** → Good  
- 🟡 **51–100** → Satisfactory  
- 🟠 **101–200** → Moderate  
- 🔴 **201–300** → Poor  
- 🟣 **301–400** → Very Poor  
- ⚫ **401–500** → Severe  
""")