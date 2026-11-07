"""Train the ice cream sales model (version 2).

Predictors: temperature (F), daylight_hours, is_weekend (0/1).
The data is simulated, to show the steps. Replace it with real sales data for a real business.
Run:  python train.py   -> writes model.pkl
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# app.py must build its input with exactly these columns, in this order
FEATURES = ["temperature", "daylight_hours", "is_weekend"]

rng = np.random.default_rng(42)
n = 700
day_of_year = rng.integers(1, 366, n)
season = np.sin((day_of_year - 80) / 365 * 2 * np.pi)          # -1 winter ... +1 summer

daylight_hours = 12 + 2.5 * season + rng.normal(0, 0.3, n)     # about 9.5 to 14.5 hours
temperature = np.clip(75 + 22 * season + rng.normal(0, 6, n), 50, 100)
is_weekend = rng.binomial(1, 2 / 7, n)

# Made-up "true" relationship: heat, long days and weekends all help
sales = -135 + 4 * temperature + 8 * daylight_hours + 30 * is_weekend + rng.normal(0, 12, n)

data = pd.DataFrame({"temperature": temperature, "daylight_hours": daylight_hours,
                     "is_weekend": is_weekend, "sales": sales})
X_train, X_test, y_train, y_test = train_test_split(data[FEATURES], data["sales"],
                                                    test_size=0.2, random_state=0)
model = LinearRegression().fit(X_train, y_train)
print("Test R^2:", round(r2_score(y_test, model.predict(X_test)), 3))
for name, coef in zip(FEATURES, model.coef_):
    print(f"  {name}: {coef:.2f}")
print("  intercept:", round(model.intercept_, 2))
print("  training means:", data[FEATURES].mean().round(2).to_dict())

# Refit on all the data for the saved model
model = LinearRegression().fit(data[FEATURES], data["sales"])
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved model.pkl")
