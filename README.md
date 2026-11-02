# Ice Cream Sales Predictor (version 1: temperature only)

Pick a city and a date. The app asks a free weather API for the forecast high, gives that
temperature to a trained model, and shows the predicted number of cones sold, with a short
business message.

## Files
- `app.py`: the Streamlit app
- `model.pkl`: the trained model (a straight-line regression), saved from Colab
- `requirements.txt`: the packages Streamlit Cloud installs for you

## How it works
1. The user picks a city and a date.
2. The app asks Open-Meteo (free, no key, no sign-up) for the forecast high for that day.
3. The model predicts sales from the temperature.
4. The app shows the number, a warning if the temperature is outside what the model learned
   from, and a short message.

The model was trained on made-up data, to show the steps. Replace it with a model trained on
real data to use this for a real business.

## Run it online for free (no terminal)
1. Put these three files in a public GitHub repository (upload them in the browser).
2. On share.streamlit.io, create an app from the repository, main file `app.py`.
No API key is needed.

## If something goes wrong
- An error when loading `model.pkl`: check that the file is in the repository next to `app.py`,
  and that `scikit-learn` is listed in `requirements.txt`.
- An error about the forecast: the weather service may be busy. Wait a minute and try again.
