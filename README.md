# Ice Cream Sales Predictor (version 3: gradient boosting model)

Pick a city and a date. The app asks a free weather API for the forecast high, gives that
temperature to a trained model, and shows the predicted number of cones sold, with a short
business message.

## Files
- `app.py`: the Streamlit app
- `train.py`: simulates the data, trains the model and saves `model.pkl` (run `python train.py`)
- `model.pkl`: the trained model (gradient boosting, chosen because it beat linear regression and random forest on unseen days)
- `model_info.json`: the model's accuracy, model comparison and predictor importance, shown in the app
- `requirements.txt`: the packages Streamlit Cloud installs for you

## How it works
1. The user picks a city and a date.
2. The app asks Open-Meteo (free, no key, no sign-up) for the forecast high, the hourly temperatures (to find the warmest time of day) and the hours of daylight for that day, and works out whether the date is a weekend.
3. The model predicts sales from temperature, daylight hours and weekend (0/1).
4. The app shows the number, a bar chart of what drove it, a warning if the temperature is outside what the model learned
   from, and a short message.

The model was trained on made-up data, to show the steps. Replace it with a model trained on
real data to use this for a real business.

## What changed in version 3
- The training data is more realistic: sales rise in an S-curve with temperature, dip in extreme
  heat, and the weekend boost is bigger on warm days.
- `train.py` now trains three models (linear regression, random forest, gradient boosting), compares
  them on days they have not seen, and saves the best one. Gradient boosting won (R² about 0.98,
  typical miss about 12 cones, versus 0.94 and 21 for the straight-line model).
- The app explains the model in the "About this model" box, and shows accuracy, the model
  comparison, predictor importance and a temperature curve in "Model accuracy and what matters".
- The data is still simulated, so these effects are built in and are not real findings.

## Run it online for free (no terminal)
1. Put these files in a public GitHub repository (upload them in the browser).
2. On share.streamlit.io, create an app from the repository, main file `app.py`.
No API key is needed.

## If something goes wrong
- An error when loading `model.pkl`: check that the file is in the repository next to `app.py`,
  and that `scikit-learn` is listed in `requirements.txt`.
- An error about the forecast: the weather service may be busy. Wait a minute and try again.
