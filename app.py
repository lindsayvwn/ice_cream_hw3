import json
import pickle
from datetime import date, datetime, timedelta

import altair as alt
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
                   "typical weekday. The bars are approximate, because the predictors also work together "
                   "(for example, a weekend helps more on a warm day).")

# ======================= Right: about the model ===========================
with side_col:
    with st.expander("About this model"):
        st.markdown(
            f"#### What it is\n"
            f"A **{info['model_name']}** model, a type of machine learning. Instead of following rules a "
            f"person wrote, it *learned* from examples: {info['n_days']:,} days, each with a temperature, "
            "hours of daylight, whether it was a weekend, and how many cones were sold.\n\n"
            "It builds many small decision trees, which are like flow charts of yes/no questions "
            "(\"Is it above 80 °F? Is it a weekend?\"). Each new tree focuses on the days the earlier "
            "trees got wrong, and the final prediction adds all the trees together.\n\n"
            "#### What it uses\n"
            "- **Forecast high temperature:** hot days make people want ice cream. Sales climb quickly as it "
            "warms up, level off once it is already hot, and dip in extreme heat.\n"
            "- **Hours of daylight:** longer days mean more time outdoors and more evening customers.\n"
            "- **Weekend (Saturday or Sunday):** more people are out, especially on warm days.\n\n"
            "#### Why this model? The business case\n"
            "We tried three kinds of model on days they had not seen: a straight-line model (linear "
            "regression), a random forest and gradient boosting. Gradient boosting predicted best, "
            f"missing by about **{info['comparison']['Gradient boosting']['test_mae']:.0f} cones** on a typical day, "
            f"against about **{info['comparison']['Linear regression']['test_mae']:.0f}** for the straight-line model.\n\n"
            "- **Accuracy is where the money is.** The prediction drives how much to stock and how many "
            "people to schedule. A smaller miss means less waste on slow days and fewer sell-outs on busy ones.\n"
            "- **It fits how sales really behave.** Sales do not rise in a straight line with heat. A "
            "straight-line model gets the hottest, busiest days wrong, and those are the days where mistakes "
            "cost the most.\n"
            "- **It is cheap to run.** The model is small and predicts instantly.\n\n"
            "**The trade-offs:** a straight-line model is easier to explain in one sentence (\"every 10 °F "
            "adds a fixed number of cones\"), it is safer with only a few weeks of data, and it keeps trending outside "
            "the temperatures it has seen. This model is harder to explain in one line, so the app shows "
            "what drove each prediction and which predictors matter most instead. It also goes flat "
            f"outside {show_temp(LOW_TEMP)} to {show_temp(HIGH_TEMP)}, so the app warns you there.\n\n"
            "#### Limits\n"
            "It was trained on made-up data to show the steps, so treat the numbers as an illustration. "
            "A real business would retrain it on its own sales records as they come in. It can only use "
            "what it was given, and it does not know about events, promotions or holidays.\n\n"
            "The *warmest time of day* comes from the hourly forecast. It does not change the number of "
            "cones predicted; it tells you when to be ready for them."
        )

    with st.expander("Model accuracy and what matters"):
        st.metric("R² (accuracy)", f"{info['r2']:.3f}",
                  help="R² is the share of the day-to-day ups and downs in sales that the model explains. "
                       "1.0 would be perfect; 0 would be no better than guessing the average every day.")
        st.markdown(
            f"**In plain terms:** the model explains about **{info['r2'] * 100:.0f}%** of the ups and downs "
            f"in daily sales, and on a typical day its guess is off by about **{info['typical_miss']:.0f} cones**. "
            f"(Measured on days the model never saw during training.)"
        )

        st.markdown("**How the models compared**")
        comparison = pd.DataFrame([
            {"Model": ("⭐ " if name == info["model_name"] else "") + name,
             "Accuracy (R²)": r["test_r2"], "Typical miss (cones)": r["test_mae"]}
            for name, r in info["comparison"].items()
        ])
        st.dataframe(comparison, hide_index=True)

        st.markdown("**How much each predictor matters**")
        imp = info["importance"]
        st.markdown(
            f"- 🌡️ **Temperature:** {imp['temperature'] * 100:.1f}%\n"
            f"- 📅 **Weekend:** {imp['is_weekend'] * 100:.1f}%\n"
            f"- ☀️ **Daylight:** {imp['daylight_hours'] * 100:.1f}%"
        )
        st.caption("Each percentage is that predictor's share of the model's predictive power, measured by how "
                   "much accuracy drops when its values are scrambled. In this made-up data, temperature is "
                   "by far the biggest driver.")

        st.markdown("**How temperature affects sales (according to the model)**")
        grid_f = list(range(50, 101, 2))
        curve = pd.DataFrame({"temperature": grid_f, "daylight_hours": 12.0})
        shown = [round(t if unit == "°F" else (t - 32) * 5 / 9, 1) for t in grid_f]
        curve_data = pd.DataFrame({
            "Weekday": model.predict(curve.assign(is_weekend=0)),
            "Weekend": model.predict(curve.assign(is_weekend=1)),
        }, index=pd.Index(shown, name=f"Forecast high ({unit})"))
        st.line_chart(curve_data, y_label="Predicted cones")
        st.caption("Notice the curve is not a straight line: sales climb quickly through the 60s and 70s, "
                   "flatten when it is hot, and the weekend boost grows on warm days. (Shown for a day with 12 "
                   "hours of daylight.)")
