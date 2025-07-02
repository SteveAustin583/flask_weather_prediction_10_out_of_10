import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# --- Load the trained artifacts ---
# These are loaded only once when the app starts
artifacts_dir = 'artifacts'
model = joblib.load(os.path.join(artifacts_dir, 'weather_prediction_model.joblib'))
label_encoder = joblib.load(os.path.join(artifacts_dir, 'weather_label_encoder.joblib'))
model_feature_names = joblib.load(os.path.join(artifacts_dir, 'model_feature_names.joblib'))

# --- How it Works: The Magic of Feature Engineering ---
# The model was trained on complex features (lags, cyclical, deltas).
# The user only provides the basic inputs.
# This function re-creates all the necessary features from the simple inputs.

def create_features_from_input(user_input):
    """
    Creates a full feature DataFrame from the user's input.
    """
    # Get current date to create cyclical features
    now = datetime.now()
    
    # Start with a dictionary of the user's direct inputs
    data = {
        'precipitation': [user_input['precipitation']],
        'temp_max': [user_input['temp_max']],
        'temp_min': [user_input['temp_min']],
        'wind': [user_input['wind']]
    }
    
    # Create cyclical features
    data['month_sin'] = [np.sin(2 * np.pi * now.month / 12)]
    data['month_cos'] = [np.cos(2 * np.pi * now.month / 12)]
    data['day_of_year_sin'] = [np.sin(2 * np.pi * now.timetuple().tm_yday / 365.25)]
    data['day_of_year_cos'] = [np.cos(2 * np.pi * now.timetuple().tm_yday / 365.25)]

    # --- Lagged and Delta Feature Creation ---
    # The most important lag feature is yesterday's weather type.
    # For a simple demo, we'll make approximations for the other lag features.
    
    # Yesterday's weather (encoded)
    yesterdays_weather_encoded = label_encoder.transform([user_input['yesterdays_weather']])[0]
    data['weather_encoded_lag1'] = [yesterdays_weather_encoded]

    # Approximate other lag/delta features. A simple approximation is to use today's
    # values for yesterday's, which makes the delta zero. This is a simplification
    # for usability, as asking the user for all of yesterday's data is complex.
    data['precipitation_lag1'] = [user_input['precipitation']]
    data['temp_max_lag1'] = [user_input['temp_max']]
    data['temp_min_lag1'] = [user_input['temp_min']]
    data['wind_lag1'] = [user_input['wind']]
    
    data['delta_temp_max'] = [0] # temp_max - temp_max_lag1
    data['delta_temp_min'] = [0] # temp_min - temp_min_lag1

    # Convert to a DataFrame and ensure the column order is correct
    df = pd.DataFrame(data)
    
    # Reorder columns to match the order the model was trained on
    return df[model_feature_names]


# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        error_message = None
        try:
            # Get data from the form and convert to float
            user_input = {
                'precipitation': float(request.form['precipitation']),
                'temp_max': float(request.form['temp_max']),
                'temp_min': float(request.form['temp_min']),
                'wind': float(request.form['wind']),
                'yesterdays_weather': request.form['yesterdays_weather']
            }
            # Check if the provided weather type is known to the encoder
            if user_input['yesterdays_weather'] not in label_encoder.classes_:
                error_message = f"Invalid input for Yesterday's Weather. Please use one of: {', '.join(label_encoder.classes_)}."
        except (ValueError, KeyError):
            # Handle cases: non-numeric input (ValueError) or missing form field (KeyError)
            error_message = "Invalid input. Please enter numbers for the numeric fields."
 
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