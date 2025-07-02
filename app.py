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

# --- UPDATED FUNCTION ---
# This function now uses the real user-provided data for yesterday to build
# the features exactly as they were built during model training.
def create_features_from_input(user_input):
    """
    Creates a full feature DataFrame from the user's input for today and yesterday.
    """
    # Get current date to create cyclical features
    now = datetime.now()
    
    # Start with a dictionary of the user's direct inputs for TODAY's weather
    data = {
        'precipitation': [user_input['precipitation']],
        'temp_max': [user_input['temp_max']],
        'temp_min': [user_input['temp_min']],
        'wind': [user_input['wind']]
    }
    
    # Create cyclical features for today
    data['month_sin'] = [np.sin(2 * np.pi * now.month / 12)]
    data['month_cos'] = [np.cos(2 * np.pi * now.month / 12)]
    data['day_of_year_sin'] = [np.sin(2 * np.pi * now.timetuple().tm_yday / 365.25)]
    data['day_of_year_cos'] = [np.cos(2 * np.pi * now.timetuple().tm_yday / 365.25)]

    # --- Lagged Feature Creation (Using REAL yesterday's data) ---
    data['precipitation_lag1'] = [user_input['yesterdays_precipitation']]
    data['temp_max_lag1'] = [user_input['yesterdays_temp_max']]
    data['temp_min_lag1'] = [user_input['yesterdays_temp_min']]
    data['wind_lag1'] = [user_input['yesterdays_wind']]
    
    # Encode yesterday's weather category
    yesterdays_weather_encoded = label_encoder.transform([user_input['yesterdays_weather']])[0]
    data['weather_encoded_lag1'] = [yesterdays_weather_encoded]

    # --- Delta Feature Creation (Using REAL today and yesterday data) ---
    # This is now a meaningful calculation instead of just being zero.
    data['delta_temp_max'] = [user_input['temp_max'] - user_input['yesterdays_temp_max']]
    data['delta_temp_min'] = [user_input['temp_min'] - user_input['yesterdays_temp_min']]

    # Convert to a DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns to match the exact order the model was trained on
    return df[model_feature_names]


# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# --- UPDATED ROUTE ---
# This route now gets all the required fields from the updated form.
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        error_message = None
        try:
            # Get data from the form and convert to float
            user_input = {
                # Today's data
                'precipitation': float(request.form['precipitation']),
                'temp_max': float(request.form['temp_max']),
                'temp_min': float(request.form['temp_min']),
                'wind': float(request.form['wind']),
                # Yesterday's data
                'yesterdays_precipitation': float(request.form['yesterdays_precipitation']),
                'yesterdays_temp_max': float(request.form['yesterdays_temp_max']),
                'yesterdays_temp_min': float(request.form['yesterdays_temp_min']),
                'yesterdays_wind': float(request.form['yesterdays_wind']),
                'yesterdays_weather': request.form['yesterdays_weather']
            }
            # Check if the provided weather type is known to the encoder
            if user_input['yesterdays_weather'] not in label_encoder.classes_:
                valid_options = ", ".join(label_encoder.classes_)
                error_message = f"Invalid input for Yesterday's Weather. Please use one of: {valid_options}."
        except (ValueError, KeyError):
            error_message = "Invalid input. Please fill out all fields with valid numbers."
 
        if error_message:
            return render_template('index.html', error=error_message)

        # Create the full feature set from user input
        features = create_features_from_input(user_input)

        # Make a prediction
        prediction_encoded = model.predict(features)[0]
        
        # Decode the prediction to get the weather name (e.g., 'sun')
        prediction_text = label_encoder.inverse_transform([prediction_encoded])[0].capitalize()

        # Render the page again with the prediction result
        return render_template('index.html', prediction=prediction_text)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)