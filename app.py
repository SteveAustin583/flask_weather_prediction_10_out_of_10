import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# --- Load the trained artifacts ---
artifacts_dir = 'artifacts'
model = joblib.load(os.path.join(artifacts_dir, 'weather_prediction_model.joblib'))
label_encoder = joblib.load(os.path.join(artifacts_dir, 'weather_label_encoder.joblib'))
model_feature_names = joblib.load(os.path.join(artifacts_dir, 'model_feature_names.joblib'))

# --- This function remains the same ---
def create_features_from_input(user_input):
    """
    Creates a full feature DataFrame from the user's input for today and yesterday.
    """
    now = datetime.now()
    data = {
        'precipitation': [user_input['precipitation']],
        'temp_max': [user_input['temp_max']],
        'temp_min': [user_input['temp_min']],
        'wind': [user_input['wind']]
    }
    data['month_sin'] = [np.sin(2 * np.pi * now.month / 12)]
    data['month_cos'] = [np.cos(2 * np.pi * now.month / 12)]
    data['day_of_year_sin'] = [np.sin(2 * np.pi * now.timetuple().tm_yday / 365.25)]
    data['day_of_year_cos'] = [np.cos(2 * np.pi * now.timetuple().tm_yday / 365.25)]
    data['precipitation_lag1'] = [user_input['yesterdays_precipitation']]
    data['temp_max_lag1'] = [user_input['yesterdays_temp_max']]
    data['temp_min_lag1'] = [user_input['yesterdays_temp_min']]
    data['wind_lag1'] = [user_input['yesterdays_wind']]
    yesterdays_weather_encoded = label_encoder.transform([user_input['yesterdays_weather']])[0]
    data['weather_encoded_lag1'] = [yesterdays_weather_encoded]
    data['delta_temp_max'] = [user_input['temp_max'] - user_input['yesterdays_temp_max']]
    data['delta_temp_min'] = [user_input['temp_min'] - user_input['yesterdays_temp_min']]
    df = pd.DataFrame(data)
    return df[model_feature_names]


# --- MODIFIED home() ROUTE ---
@app.route('/')
def home():
    # Pass the available weather options to the template for the dropdown
    weather_options = label_encoder.classes_
    return render_template('index.html', weather_options=weather_options)


# --- MODIFIED predict() ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    # Get the weather options to pass back to the template in case of error or success
    weather_options = label_encoder.classes_

    if request.method == 'POST':
        error_message = None
        try:
            user_input = {
                'precipitation': float(request.form['precipitation']),
                'temp_max': float(request.form['temp_max']),
                'temp_min': float(request.form['temp_min']),
                'wind': float(request.form['wind']),
                'yesterdays_precipitation': float(request.form['yesterdays_precipitation']),
                'yesterdays_temp_max': float(request.form['yesterdays_temp_max']),
                'yesterdays_temp_min': float(request.form['yesterdays_temp_min']),
                'yesterdays_wind': float(request.form['yesterdays_wind']),
                'yesterdays_weather': request.form['yesterdays_weather']
            }
            if user_input['yesterdays_weather'] not in weather_options:
                valid_options = ", ".join(weather_options)
                error_message = f"Invalid input for Yesterday's Weather. Please use one of: {valid_options}."
        except (ValueError, KeyError):
            error_message = "Invalid input. Please fill out all fields with valid numbers."
 
        if error_message:
            # Pass the options back to the template along with the error
            return render_template('index.html', error=error_message, weather_options=weather_options)

        # Create the full feature set from user input
        features = create_features_from_input(user_input)

        # Make a prediction
        prediction_encoded = model.predict(features)[0]
        prediction_text = label_encoder.inverse_transform([prediction_encoded])[0].capitalize()

        # Pass the options back to the template along with the prediction
        return render_template('index.html', prediction=prediction_text, weather_options=weather_options)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)