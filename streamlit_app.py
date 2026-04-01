import streamlit as st
import pandas as pd
import joblib
import requests
import sys
import os

# ---------------- IMPORT CUSTOM ----------------
sys.path.append(os.path.abspath("."))
from src.aqi_utils import get_aqi_category, get_health_advisory

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AQI Dashboard", layout="centered")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Data/cleaned_air_quality.csv")
st.write(os.path.abspath("Data/cleaned_air_quality.csv"))
model = joblib.load("Model/aqi_model.pkl")

# ---------------- LIVE AQI ----------------
def get_live_aqi(city):
    try:
        url = f"https://api.waqi.info/feed/{city}/?token=demo"
        res = requests.get(url)
        data = res.json()
        if data["status"] == "ok":
            return data["data"]["aqi"]
        return None
    except:
        return None

# ---------------- TITLE ----------------
st.markdown("<h2 style='text-align:center;'>Air Quality Intelligence Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Real-time monitoring with AI-based prediction</p>", unsafe_allow_html=True)

# ---------------- NAVIGATION ----------------
menu = st.sidebar.radio("Navigation", ["Dashboard", "Insights", "Trend"])

# =====================================================
# DASHBOARD
# =====================================================
if menu == "Dashboard":

    st.markdown("### Select Location")

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", sorted(df["City"].unique()))

    with col2:
        year = st.selectbox("Year", sorted(df["Year"].unique()))

    st.markdown("---")

    # LIVE AQI
    live_aqi = get_live_aqi(city.lower())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Live AQI")
        if live_aqi:
            st.metric("Current AQI", live_aqi)
        else:
            st.warning("Live data unavailable")

    # PREDICTED AQI
    filtered = df[(df["City"] == city) & (df["Year"] == year)]

    if not filtered.empty:

        avg_values = filtered[["PM2.5","PM10","NO2","NH3","CO","SO2","O3"]].mean()
        predicted = model.predict([avg_values])[0]
        predicted = round(predicted, 2)

        with col2:
            st.markdown("#### Predicted AQI")
            st.metric("Model Prediction", predicted)

        # CATEGORY + ADVISORY
        category = get_aqi_category(predicted)
        advisory = get_health_advisory(category)

        # COLOR FUNCTION
        def get_color(aqi):
            if aqi <= 50:
                return "green"
            elif aqi <= 100:
                return "lime"
            elif aqi <= 200:
                return "orange"
            elif aqi <= 300:
                return "red"
            else:
                return "purple"

        color = get_color(predicted)

        # BIG AQI CARD
        st.markdown(f"""
        <div style="
        background-color:#222;
        padding:20px;
        border-radius:12px;
        text-align:center;
        margin-top:15px;">
        <h2 style="color:{color}; margin:0;">{predicted}</h2>
        <p style="color:white;">Air Quality Index</p>
        </div>
        """, unsafe_allow_html=True)

        # COMPARISON
        st.markdown("### Analysis")

        if live_aqi:
            diff = live_aqi - predicted
            if diff > 0:
                st.error(f"Air quality is deteriorating by {abs(diff):.2f} AQI points")
            else:
                st.success(f"Air quality is improving by {abs(diff):.2f} AQI points")

        # HEALTH INSIGHT
        st.markdown("### Health Insight")

        st.markdown(f"""
        <div style='padding:15px;border-radius:10px;background-color:#222;'>
            <h4 style="color:{color};">Category: {category}</h4>
            <p style="color:white;">{advisory}</p>
        </div>
        """, unsafe_allow_html=True)

        # SMART LINE
        st.markdown("""
        <p style='color:gray; font-size:13px; text-align:center; margin-top:10px;'>
        This system combines real-time environmental data with machine learning predictions to provide intelligent air quality insights.
        </p>
        """, unsafe_allow_html=True)

    else:
        st.warning("No data available")

# =====================================================
# INSIGHTS
# =====================================================
elif menu == "Insights":

    st.markdown("### Top Polluted Cities")

    top = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(5)
    st.bar_chart(top)

# =====================================================
# TREND
# =====================================================
elif menu == "Trend":

    st.markdown("### AQI Trend Analysis")

    city = st.selectbox("Select City", sorted(df["City"].unique()))
    trend = df[df["City"] == city].groupby("Year")["AQI"].mean()

    st.line_chart(trend)
    