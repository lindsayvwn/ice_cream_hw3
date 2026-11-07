# Ice Cream Sales Predictor (version 2: temperature, daylight, weekend)

Pick a city and a date. The app asks a free weather API for the forecast high, gives that
temperature to a trained model, and shows the predicted number of cones sold, with a short
business message.

## Files
- `app.py`: the Streamlit app
- `train.py`: simulates the data, trains the model and saves `model.pkl` (run `python train.py`)
- `model.pkl`: the trained model (a multiple linear regression)
- `requirements.txt`: the packages Streamlit Cloud installs for you

## How it works
1. The user picks a city and a date.
2. The app asks Open-Meteo (free, no key, no sign-up) for the forecast high and the hours of daylight for that day, and works out whether the date is a weekend.
3. The model predicts sales from temperature, daylight hours and weekend (0/1).
4. The app shows the number, a bar chart of what drove it, a warning if the temperature is outside what the model learned
   from, and a short message.

The model was trained on made-up data, to show the steps. Replace it with a model trained on
real data to use this for a real business.

## What changed in version 2
Added two predictors to temperature: **daylight_hours** (long summer days mean more time outside)
and **is_weekend** (more foot traffic on Saturday/Sunday). The app now also shows an explanation box
of the predictors and a bar chart of how much each one moved the prediction. The effects were built
into the made-up training data, so they are not real findings.

## Run it online for free (no terminal)
1. Put these three files in a public GitHub repository (upload them in the browser).
2. On share.streamlit.io, create an app from the repository, main file `app.py`.
No API key is needed.

## If something goes wrong
- An error when loading `model.pkl`: check that the file is in the repository next to `app.py`,
  and that `scikit-learn` is listed in `requirements.txt`.
- An error about the forecast: the weather service may be busy. Wait a minute and try again.
