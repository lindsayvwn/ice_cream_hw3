import json
import pickle
import time
from datetime import date, datetime, timedelta

import altair as alt
import pandas as pd
import requests
import streamlit as st

# ---- Settings ----------------------------------------------------------
CITIES = {
    "Pittsburgh": (40.44, -79.99),
    "New York": (40.71, -74.01),
    "Boston": (42.36, -71.06),
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


class ForecastError(Exception):
    """Raised when the weather service can't give us a usable forecast."""


@st.cache_data(ttl=1800, show_spinner="Getting the forecast...")
def fetch_forecast(latitude, longitude, day):
    """Ask Open-Meteo for one day's forecast (always Fahrenheit: that is what the model learned).
    Results are cached for 30 minutes so repeat requests don't hit the service again."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,daylight_duration",
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "auto",
        "start_date": day,
        "end_date": day,
    }
    response = None
    for attempt in range(3):                       # retry if the service is busy or briefly down
        try:
            response = requests.get(URL, params=params, timeout=30)
        except requests.RequestException:
            response = None
        else:
            if response.status_code not in (429, 500, 502, 503, 504):
                break
        if attempt < 2:
            time.sleep(1.5 * (attempt + 1))
    if response is None:
        raise ForecastError("Could not reach the weather service. Please try again in a moment.")
    if response.status_code == 429:
        raise ForecastError("The weather service is busy right now (too many requests). "
                            "Please wait a minute and try again.")
    if not response.ok:
        try:
            reason = response.json().get("reason", "")
        except ValueError:
            reason = ""
        raise ForecastError(f"The weather service could not give a forecast for that date "
                            f"(error {response.status_code}). {reason}".strip())
    data = response.json()
    try:
        temp = data["daily"]["temperature_2m_max"][0]
        daylight = data["daily"]["daylight_duration"][0]
        hourly_times, hourly_temps = data["hourly"]["time"], data["hourly"]["temperature_2m"]
    except (KeyError, IndexError):
        raise ForecastError("The weather service sent back an unexpected response. Please try again.")
    if temp is None or daylight is None or any(t is None for t in hourly_temps):
        raise ForecastError("The forecast for that date is not available yet. Try an earlier date.")
    return {"temperature": temp, "daylight_seconds": daylight,
            "hourly_times": hourly_times, "hourly_temps": hourly_temps}


def hour_label(hour):
    """13 -> '1 PM'"""
    return datetime(2000, 1, 1, hour).strftime("%I %p").lstrip("0")


st.set_page_config(page_title="Ice Cream Sales Predictor", page_icon="🍦")
st.title("🍦 Ice Cream Sales Predictor")
st.caption("Pick a city and a date and the app will predict how many cones you will sell.")

# Make the tab labels bigger
st.markdown(
    """
    <style>
    [role="tablist"] { gap: 12px; }
    [role="tab"] { height: 3.4rem; padding: 0 1.4rem; }
    [role="tab"] p { font-size: 1.4rem; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

predict_tab, model_tab = st.tabs(["Predict", "Model information"])

# ======================= Tab 1: the app ===================================
with predict_tab:
    # One compact panel for all the controls
    with st.container(border=True):
        city_col, date_col, unit_col = st.columns([3, 3, 2])
        city = city_col.selectbox("City", list(CITIES))
        selected_date = date_col.date_input(
            "Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=15),    # the forecast covers today plus 15 more days
        )
        unit = unit_col.segmented_control("Temperature unit", ["°F", "°C"], default="°F") or "°F"   # None if unselected
        predict_clicked = st.button("Predict sales", type="primary")

    def show_temp(fahrenheit):
        """Convert a Fahrenheit value for display in the chosen unit."""
        value = fahrenheit if unit == "°F" else (fahrenheit - 32) * 5 / 9
        return f"{value:.0f} {unit}"

    if predict_clicked:
        # 1. Ask the weather API for the forecast
        latitude, longitude = CITIES[city]
        try:
            data = fetch_forecast(latitude, longitude, str(selected_date))
        except ForecastError as error:
            st.session_state.pop("result", None)
            st.error(str(error))
        else:
            hourly_temps = data["hourly_temps"]
            peak_index = max(range(len(hourly_temps)), key=lambda i: hourly_temps[i])

            # Keep the result in session_state so it stays on screen when the unit is switched
            st.session_state["result"] = {
                "temperature": data["temperature"],
                "daylight_hours": data["daylight_seconds"] / 3600,            # API gives seconds
                "is_weekend": int(selected_date.weekday() >= 5),              # Saturday = 5, Sunday = 6
                "peak_hour": int(data["hourly_times"][peak_index][11:13]),    # local time, e.g. "2026-11-08T15:00"
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

        # 5. Chart: how much each predictor moved the prediction away from a typical weekday.
        #    For each predictor, swap just that one back to its typical value and see how the prediction changes.
        effects = {}
        for column, label in [("temperature", "Temperature"), ("daylight_hours", "Daylight"),
                              ("is_weekend", "Weekend")]:
            typical_version = features.copy()
            typical_version[column] = TYPICAL_DAY[column]
            effects[label] = prediction - model.predict(typical_version)[0]
        baseline = model.predict(pd.DataFrame([TYPICAL_DAY]))[0]
        chart_data = pd.DataFrame({
            "Predictor": [f"🌡️ Temperature ({show_temp(temperature)})",
                          f"☀️ Daylight ({daylight_hours:.1f} h)",
                          "📅 Weekend" if is_weekend else "📅 Weekday"],
            "Effect": list(effects.values()),
        })
        chart_data["Direction"] = chart_data["Effect"].apply(lambda v: "Adds cones" if v >= 0 else "Removes cones")
        chart_data["Label"] = chart_data["Effect"].apply(lambda v: f"{v:+.0f}")

        st.subheader("What is driving this prediction?")
        st.markdown(f"A typical weekday ({show_temp(TYPICAL_DAY['temperature'])}, "
                    f"{TYPICAL_DAY['daylight_hours']} h of daylight) sells about **{baseline:.0f} cones**. "
                    f"Here is what moves **{city} on {selected_date:%b %d}** to **{prediction:.0f}**:")

        chart_data = chart_data.reindex(chart_data["Effect"].abs().sort_values(ascending=False).index)   # biggest first
        low = min(0, chart_data["Effect"].min()) * 1.35
        high = max(0, chart_data["Effect"].max()) * 1.35
        low, high = (low, high) if high - low >= 20 else (low - 10, high + 10)
        base = alt.Chart(chart_data).encode(
            y=alt.Y("Predictor:N", sort=list(chart_data["Predictor"]), title=None, axis=alt.Axis(labelLimit=260, labelFontSize=14, labelColor="#333", ticks=False, domain=False, labelPadding=10)),
        )
        bars = base.mark_bar(cornerRadiusEnd=5, size=30).encode(
            x=alt.X("Effect:Q", title="Cones added or removed", scale=alt.Scale(domain=[low, high], nice=False),
                    axis=alt.Axis(grid=True, gridColor="#efefef", tickCount=5, domain=False, labelColor="#888", titleColor="#888", titleFontWeight="normal")),
            color=alt.Color("Direction:N", legend=None,
                            scale=alt.Scale(domain=["Adds cones", "Removes cones"], range=["#e07a5f", "#5b8db8"])),
            tooltip=[alt.Tooltip("Predictor:N"), alt.Tooltip("Effect:Q", format="+.1f", title="Cones")],
        )
        # put each label just beyond the end of its bar (right of positive bars, left of negative ones)
        label_pos = base.mark_text(align="left", dx=6, fontSize=15, fontWeight="bold", color="#333").encode(
            x="Effect:Q", text="Label:N").transform_filter(alt.datum.Effect >= 0)
        label_neg = base.mark_text(align="right", dx=-6, fontSize=15, fontWeight="bold", color="#333").encode(
            x="Effect:Q", text="Label:N").transform_filter(alt.datum.Effect < 0)
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#888", strokeWidth=1.5).encode(x="x:Q")
        chart = (bars + label_pos + label_neg + zero).properties(height=190, width="container").configure_view(strokeWidth=0)
        st.altair_chart(chart)
        st.caption("Each bar is how many cones that predictor adds (orange) or removes (blue) compared with a "
                   "typical weekday. Added together, the three bars take you from the typical weekday to this "
                   "prediction.")

# ======================= Tab 2: model information =========================
with model_tab:
    with st.expander("About this model", expanded=True):
        coefs = info["coefficients"]
        st.markdown(
            f"#### What it is\n"
            f"A **linear regression** model, one of the simplest and most widely used kinds of machine "
            f"learning. Instead of following rules a person wrote, it *learned* from examples: "
            f"{info['n_days']:,} days, each with a temperature, hours of daylight, whether it was a weekend, "
            "and how many cones were sold.\n\n"
            "It finds the best \"recipe\" for predicting sales by adding up weighted inputs:\n\n"
            "> predicted cones = starting point + (a × temperature) + (b × hours of daylight) + (c × weekend)\n\n"
            "Training means trying many values for a, b and c and keeping the ones that make the predictions "
            "closest to the real sales (this is called least squares). Those learned numbers are the "
            "**coefficients**, and you can see them in the next section.\n\n"
            "#### What it uses\n"
            "- **Forecast high temperature:** hot days make people want ice cream.\n"
            "- **Hours of daylight:** longer days mean more time outdoors and more evening customers.\n"
            "- **Weekend (Saturday or Sunday):** more people are out, so foot traffic is higher.\n\n"
            "#### Why this model? The business case\n"
            "- **You can explain it.** Every predictor has one number you can read and check against your "
            "own sense of the business (\"every 10 °F warmer adds about "
            f"{coefs['temperature'] * 10:.0f} cones\"). That makes it easy to trust and easy to defend.\n"
            "- **It is safe with limited data.** A simple model is less likely to chase noise when you only "
            "have a few weeks of sales.\n"
            "- **It is cheap to run.** It predicts instantly, and it keeps trending sensibly even outside the "
            "temperatures it has seen.\n\n"
            "**The trade-off:** it assumes each predictor adds a fixed amount of cones, a straight-line "
            "effect. Real sales curve: they climb fast as it warms up, level off when it is already hot, and "
            "the weekend boost is bigger on warm days. So it can miss on the hottest and coldest days. "
            "More flexible models can fit those curves, but they are harder to explain.\n\n"
            "#### Limits\n"
            "It was trained on made-up data to show the steps, so treat the numbers as an illustration. "
            "A real business would retrain it on its own sales records as they come in. It can only use "
            "what it was given, and it does not know about events, promotions or holidays. It was trained on "
            f"{show_temp(LOW_TEMP)} to {show_temp(HIGH_TEMP)}, so the app warns you outside that range.\n\n"
            "The *warmest time of day* comes from the hourly forecast. It does not change the number of "
            "cones predicted; it tells you when to be ready for them."
        )

    with st.expander("Model accuracy and coefficients"):
        st.metric("R² (accuracy)", f"{info['r2']:.3f}",
                  help="R² is the share of the day-to-day ups and downs in sales that the model explains. "
                       "1.0 would be perfect; 0 would be no better than guessing the average every day.")
        st.markdown(
            f"**In plain terms:** the model explains about **{info['r2'] * 100:.0f}%** of the ups and downs "
            f"in daily sales, and on a typical day its guess is off by about **{info['typical_miss']:.0f} cones**. "
            f"(Measured on days the model never saw during training.)"
        )

        st.markdown("**What each predictor is worth**")
        per_10 = coefs["temperature"] * 10 * (1 if unit == "°F" else 9 / 5)
        st.markdown(
            f"- 🌡️ **Temperature:** every **10 {unit}** warmer adds about **{per_10:.0f} cones**.\n"
            f"- ☀️ **Daylight:** every extra **hour of daylight** adds about **{coefs['daylight_hours']:.0f} cones**.\n"
            f"- 📅 **Weekend:** a Saturday or Sunday sells about **{coefs['is_weekend']:.0f} more cones** "
            f"than a weekday."
        )
        st.caption("These are the model's coefficients. Each one is the effect of that predictor "
                   "with the other two held the same.")

        coef_table = pd.DataFrame({
            "Predictor": [f"Temperature (per {unit})", "Daylight (per hour)", "Weekend (vs. weekday)"],
            "Coefficient (cones)": [round(coefs["temperature"] * (1 if unit == "°F" else 9 / 5), 2),
                                    coefs["daylight_hours"], coefs["is_weekend"]],
        })
        st.dataframe(coef_table, hide_index=True)
        st.caption(f"Raw numbers. Intercept (the model's starting point before adding the predictors): "
                   f"{info['intercept']:.1f}. Trained on {info['n_days']:,} simulated days.")
