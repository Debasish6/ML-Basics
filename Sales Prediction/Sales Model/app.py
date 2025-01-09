from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = joblib.load('sales_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

numerical_features = ['Month', 'Year', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from the request
    df = pd.DataFrame(data)
    
    df = pd.get_dummies(df, columns=['DEALSIZE'], drop_first=True)
    
    # Align columns
    expected_columns = model.feature_names_in_  # Adjust this according to your model's feature set
    missing_cols = set(expected_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[expected_columns]
    
    df[numerical_features] = scaler.transform(df[numerical_features])
    
    predictions = model.predict(df)
    
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
