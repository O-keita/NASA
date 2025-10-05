from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'xgb_aqi_no2.joblib')

model = joblib.load(model_path)


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
    app.run(debug=True, host='0.0.0.0', port=7000)