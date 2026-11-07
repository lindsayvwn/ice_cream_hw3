import json
import pickle
from datetime import date, datetime, timedelta

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

# Load the trained model and its summary numbers (both written by train.py)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model_info.json") as f:
    info = json.load(f)


def hour_label(hour):
    """13 -> '1 PM'"""
    return datetime(2000, 1, 1, hour).strftime("%I %p").lstrip("0")


st.set_page_config(page_title="Ice Cream Sales Predictor", page_icon="🍦", layout="wide")
st.title("🍦 Ice Cream Sales Predictor")
st.caption("Pick a city and a date and the app will predict how many cones you will sell.")

main_col, side_col = st.columns([2, 1], gap="large")

# ======================= Left: the app ====================================
with main_col:
    unit = st.radio("Temperature unit", ["°F", "°C"], horizontal=True)

    def show_temp(fahrenheit):
        """Convert a Fahrenheit value for display in the chosen unit."""
        value = fahrenheit if unit == "°F" else (fahrenheit - 32) * 5 / 9
        return f"{value:.0f} {unit}"

    city = st.selectbox("City", list(CITIES))
    selected_date = st.date_input(
        "Date",
        value=date.today() + timedelta(days=1),
        min_value=date.today(),
        max_value=date.today() + timedelta(days=16),    # the forecast reaches about 16 days ahead
    )

    if st.button("Predict sales", type="primary"):
        # 1. Ask the weather API for the forecast (always in Fahrenheit: that is what the model learned)
        latitude, longitude = CITIES[city]
        response = requests.get(
            URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "temperature_2m_max,daylight_duration",
                "hourly": "temperature_2m",
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
                "start_date": str(selected_date),
                "end_date": str(selected_date),
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        hourly_times = data["hourly"]["time"]            # local times like "2026-11-08T15:00"
        hourly_temps = data["hourly"]["temperature_2m"]
        peak_index = max(range(len(hourly_temps)), key=lambda i: hourly_temps[i])

        # Keep the result in session_state so it stays on screen when the unit is switched
        st.session_state["result"] = {
            "temperature": data["daily"]["temperature_2m_max"][0],
            "daylight_hours": data["daily"]["daylight_duration"][0] / 3600,   # API gives seconds
            "is_weekend": int(selected_date.weekday() >= 5),                  # Saturday = 5, Sunday = 6
            "peak_hour": int(hourly_times[peak_index][11:13]),
        }

    result = st.session_state.get("result")
    if result:
        temperature = result["temperature"]
        daylight_hours = result["daylight_hours"]
        is_weekend = result["is_weekend"]
        peak_hour = result["peak_hour"]

        # 2. Give the model the same input it was trained on, then predict
        features = pd.DataFrame({
            "temperature": [temperature],
            "daylight_hours": [daylight_hours],
            "is_weekend": [is_weekend],
        })
        prediction = max(0, model.predict(features)[0])

        # 3. Show the result
        col1, col2, col3 = st.columns(3)
        col1.metric("Forecast high", show_temp(temperature))
        col2.metric("Daylight", f"{daylight_hours:.1f} hours")
        col3.metric("Predicted sales", f"{prediction:.0f} cones")
        st.caption("Weekend" if is_weekend else "Weekday")

        # 4. When will it be warmest? Plan staffing and stock around it.
        rush_start, rush_end = max(0, peak_hour - 2), min(23, peak_hour + 2)
        st.success(
            f"🌡️ It will be warmest around **{hour_label(peak_hour)}** ({show_temp(temperature)}). "
            f"Plan accordingly: have the freezers full and enough staff on by {hour_label(rush_start)}, "
            f"because the busiest stretch is likely {hour_label(rush_start)} to {hour_label(rush_end)}."
        )

        if temperature < LOW_TEMP or temperature > HIGH_TEMP:
            st.warning(f"This temperature is outside the range the model learned from "
                       f"({show_temp(LOW_TEMP)} to {show_temp(HIGH_TEMP)}). Treat the number with caution.")

        if prediction > HIGH_SALES:
            st.info("Busy day: make sure you are fully stocked.")
        elif prediction > LOW_SALES:
            st.info("A normal day is expected.")
        else:
            st.info("A slow day: consider a promotion, or order less.")

        # 5. Chart: how much each predictor moved the prediction away from a typical day
        typical = pd.Series(TYPICAL_DAY)
        effects = pd.Series(model.coef_, index=features.columns) * (features.iloc[0] - typical)
        chart_data = pd.DataFrame({
            "Predictor": ["Temperature", "Daylight", "Weekend"],
            "Cones added or removed": effects[["temperature", "daylight_hours", "is_weekend"]].values,
        }).set_index("Predictor")
        st.subheader("What is driving this prediction?")
        st.bar_chart(chart_data)
        st.caption(f"Each bar shows how many more (or fewer) cones this predictor adds compared with a typical "
                   f"weekday ({show_temp(TYPICAL_DAY['temperature'])}, {TYPICAL_DAY['daylight_hours']} hours of daylight).")

# ======================= Right: about the model ===========================
with side_col:
    with st.expander("About this model"):
        st.markdown(
            "The model predicts daily cone sales from three things:\n"
            "- **Forecast high temperature:** hot days make people want ice cream. This is the biggest driver.\n"
            "- **Hours of daylight:** longer days (late spring and summer) mean more time outdoors and more "
            "evening customers, and daylight also captures the season.\n"
            "- **Weekend (Saturday or Sunday):** more people are off work and out with family, so foot traffic is higher.\n\n"
            "It is a linear regression, so each predictor adds or removes a fixed number of cones. "
            "It was trained on made-up data to show the steps, so treat the numbers as an illustration.\n\n"
            "The *warmest time of day* comes from the hourly forecast. It does not change the number of cones "
            "predicted; it tells you when to be ready for them."
        )

    st.metric("R² (on held-out test data)", f"{info['r2']:.3f}",
              help="Share of the day-to-day variation in sales that the model explains. 1.0 is perfect, 0 is no better than the average.")

    temp_coef = info["coefficients"]["temperature"] * (1 if unit == "°F" else 9 / 5)
    coef_table = pd.DataFrame({
        "Predictor": [f"Temperature (per {unit})", "Daylight (per hour)", "Weekend (vs. weekday)"],
        "Coefficient (cones)": [round(temp_coef, 2), info["coefficients"]["daylight_hours"],
                                info["coefficients"]["is_weekend"]],
    })
    st.markdown("**Coefficients**")
    st.dataframe(coef_table, hide_index=True)
    st.caption(f"Intercept: {info['intercept']:.1f} cones. Each coefficient is how many more cones are "
               f"sold when that predictor goes up by one unit, with the others held the same. "
               f"Trained on {info['n_days']} simulated days.")
