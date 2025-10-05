from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import joblib

app = Flask(__name__)

# Load the model (use 'joblib' if you saved with joblib)
model = joblib.load('xgb_aqi_no2.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Expecting: {"no2": 99.88}
    no2 = data.get('no2')
    if no2 is None:
        return jsonify({'error': 'Missing NO2 value'}), 400
    # Prepare input for prediction
    arr = np.array([[no2]])
    pred = model.predict(arr)[0]
    return jsonify({'aqi_predicted': float(pred), 'no2': float(no2)})

if __name__ == '__main__':
    app.run(debug=True)