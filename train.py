"""Train the ice cream sales model.

A linear regression on three predictors: temperature (F), daylight_hours, is_weekend (0/1).
The data is simulated, to show the steps. Replace it with real sales data for a real business.
Run:  python train.py   -> writes model.pkl and model_info.json
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# app.py must build its input with exactly these columns, in this order
FEATURES = ["temperature", "daylight_hours", "is_weekend"]

# ---- 1. Simulate the data -------------------------------------------------
rng = np.random.default_rng(42)
n = 1500
day_of_year = rng.integers(1, 366, n)
season = np.sin((day_of_year - 80) / 365 * 2 * np.pi)          # -1 winter ... +1 summer

daylight_hours = 12 + 2.5 * season + rng.normal(0, 0.3, n)     # about 9.5 to 14.5 hours
temperature = np.clip(75 + 22 * season + rng.normal(0, 6, n), 50, 100)
is_weekend = rng.binomial(1, 2 / 7, n)

# Made-up "true" relationship, closer to real life than a straight line:
#  - sales climb steeply as it warms up, then level off once it is already hot (S-curve)
#  - extreme heat (above 92 F) keeps people at home, so sales dip
#  - weekends help, and they help MORE on warm days (families go out for ice cream)
warm = 1 / (1 + np.exp(-(temperature - 70) / 6))
sales = (60 + 230 * warm
         - 0.15 * np.clip(temperature - 92, 0, None) ** 2
         + 6 * (daylight_hours - 12)
         + is_weekend * (10 + 30 * warm)
         + rng.normal(0, 15, n))

data = pd.DataFrame({"temperature": temperature, "daylight_hours": daylight_hours,
                     "is_weekend": is_weekend, "sales": sales})
X, y = data[FEATURES], data["sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ---- 2. Fit the model and check it on days it has not seen ----------------
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
print(f"Test R2 {r2:.3f} | typical miss {mae:.1f} cones")
for name, coef in zip(FEATURES, model.coef_):
    print(f"  {name}: {coef:.2f}")
print("  intercept:", round(model.intercept_, 2))

# ---- 3. Refit on all the data and save it ---------------------------------
final_model = LinearRegression().fit(X, y)
with open("model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("model_info.json", "w") as f:
    json.dump({"model_name": "Linear regression", "r2": round(r2, 3), "typical_miss": round(mae, 1),
               "n_days": n,
               "coefficients": dict(zip(FEATURES, [round(float(c), 2) for c in final_model.coef_])),
               "intercept": round(float(final_model.intercept_), 2)}, f, indent=2)
print("Saved model.pkl and model_info.json")
