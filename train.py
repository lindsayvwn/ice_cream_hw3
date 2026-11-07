"""Train the ice cream sales model (version 3).

Predictors: temperature (F), daylight_hours, is_weekend (0/1).
The data is simulated, to show the steps. Replace it with real sales data for a real business.

Several kinds of model compete on days they have never seen, and the best one is saved.
Run:  python train.py   -> writes model.pkl and model_info.json
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

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

# ---- 2. Let several models compete ---------------------------------------
candidates = {
    "Linear regression": LinearRegression(),
    "Random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0),
    "Gradient boosting": GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05,
                                                   subsample=0.8, random_state=0),
}
results = {}
for name, candidate in candidates.items():
    cv_r2 = cross_val_score(candidate, X_train, y_train, cv=5, scoring="r2").mean()
    candidate.fit(X_train, y_train)
    pred = candidate.predict(X_test)
    results[name] = {"cv_r2": round(cv_r2, 3), "test_r2": round(r2_score(y_test, pred), 3),
                     "test_mae": round(mean_absolute_error(y_test, pred), 1)}
    print(f"{name:18s} CV R2 {cv_r2:.3f} | test R2 {results[name]['test_r2']:.3f} | "
          f"typical miss {results[name]['test_mae']:.1f} cones")

# Pick the winner using cross-validation on the training days only (the test days stay untouched)
best_name = max(results, key=lambda k: results[k]["cv_r2"])
print("Best model:", best_name)

# ---- 3. How much does each predictor matter? ------------------------------
best = candidates[best_name]
imp = permutation_importance(best, X_test, y_test, n_repeats=20, random_state=0, scoring="r2")
share = np.clip(imp.importances_mean, 0, None)
share = share / share.sum()
importance = {f: round(float(s), 3) for f, s in zip(FEATURES, share)}
print("Importance:", importance)

# ---- 4. Refit the winner on all the data and save it ----------------------
final_model = type(best)(**best.get_params()).fit(X, y)
with open("model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("model_info.json", "w") as f:
    json.dump({"model_name": best_name, "r2": results[best_name]["test_r2"],
               "typical_miss": results[best_name]["test_mae"], "n_days": n,
               "comparison": results, "importance": importance}, f, indent=2)
print("Saved model.pkl and model_info.json")
