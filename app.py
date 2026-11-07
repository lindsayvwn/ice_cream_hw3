import pickle
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

# ---- Settings ----------------------------------------------------------
CITIES = {
    "Pittsburgh": (40.44, -79.99),
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Miami": (25.76, -80.19),
    "Los Angeles": (34.05, -118.24),
}
LOW_TEMP, HIGH_TEMP = 50, 100        # the temperatures (F) the model learned from
LOW_SALES, HIGH_SALES = 200, 300     # below LOW_SALES is a slow day, above HIGH_SALES is a busy one
TYPICAL_DAY = {"temperature": 75, "daylight_hours": 12, "is_weekend": 0}   # baseline for the chart
URL = "https://api.open-meteo.com/v1/forecast"
# ------------------------------------------------------------------------

# Load the trained model (we trained it once, in Colab, and saved it to a file)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Ice Cream Sales Predictor", page_icon="🍦")
st.title("🍦 Ice Cream Sales Predictor")
st.caption("Pick a city and a date. The app gets the weather forecast, checks whether it is a weekend, and predicts how many cones you will sell.")

with st.container(border=True):
    st.markdown("**About this model: what it uses and why**")
    st.markdown(
        "The model predicts daily cone sales from three things:\n"
        "- **Forecast high temperature (°F):** hot days make people want ice cream. This is the biggest driver.\n"
        "- **Hours of daylight:** longer days (late spring and summer) mean more time outdoors and more "
        "evening customers, and daylight also captures the season.\n"
        "- **Weekend (Saturday or Sunday):** more people are off work and out with family, so foot traffic is higher.\n\n"
        "It is a linear regression, so each predictor adds or removes a fixed number of cones. "
        "It was trained on made-up data to show the steps, so treat the numbers as an illustration."
    )

city = st.selectbox("City", list(CITIES))
selected_date = st.date_input(
    "Date",
    value=date.today() + timedelta(days=1),
    min_value=date.today(),
    max_value=date.today() + timedelta(days=16),    # the forecast reaches about 16 days ahead
)

if st.button("Predict sales", type="primary"):
    # 1. Ask the weather API for the forecast
    latitude, longitude = CITIES[city]
    response = requests.get(
        URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,daylight_duration",
            "temperature_unit": "fahrenheit",
            "timezone": "auto",
            "start_date": str(selected_date),
            "end_date": str(selected_date),
        },
        timeout=30,
    )
    response.raise_for_status()
    daily = response.json()["daily"]
    temperature = daily["temperature_2m_max"][0]
    daylight_hours = daily["daylight_duration"][0] / 3600   # API gives seconds
    is_weekend = int(selected_date.weekday() >= 5)          # Saturday = 5, Sunday = 6

    # 2. Give the model the same input it was trained on, then predict
    features = pd.DataFrame({
        "temperature": [temperature],
        "daylight_hours": [daylight_hours],
        "is_weekend": [is_weekend],
    })
    prediction = max(0, model.predict(features)[0])

    # 3. Show the result
    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast high", f"{temperature:.0f} °F")
    col2.metric("Daylight", f"{daylight_hours:.1f} hours")
    col3.metric("Predicted sales", f"{prediction:.0f} cones")
    st.caption("Weekend" if is_weekend else "Weekday")

    # 4. Chart: how much each predictor moved the prediction away from a typical day
    typical = pd.Series(TYPICAL_DAY)
    today = features.iloc[0]
    effects = pd.Series(model.coef_, index=features.columns) * (today - typical)
    chart_data = pd.DataFrame({
        "Predictor": ["Temperature", "Daylight", "Weekend"],
        "Cones added or removed": effects[["temperature", "daylight_hours", "is_weekend"]].values,
    }).set_index("Predictor")
    st.subheader("What is driving this prediction?")
    st.bar_chart(chart_data)
    st.caption(f"Each bar shows how many more (or fewer) cones this predictor adds compared with a typical "
               f"weekday ({TYPICAL_DAY['temperature']} °F, {TYPICAL_DAY['daylight_hours']} hours of daylight).")

    if temperature < LOW_TEMP or temperature > HIGH_TEMP:
        st.warning(f"This temperature is outside the range the model learned from ({LOW_TEMP} to {HIGH_TEMP} °F). "
                   "Treat the number with caution.")

    if prediction > HIGH_SALES:
        st.info("Busy day: make sure you are fully stocked.")
    elif prediction > LOW_SALES:
        st.info("A normal day is expected.")
    else:
        st.info("A slow day: consider a promotion, or order less.")
