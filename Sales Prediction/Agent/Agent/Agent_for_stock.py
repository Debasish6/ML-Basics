import pandas as pd
from langchain.chains import SequentialChain
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


data = {
    'StoreID': [1, 1, 2, 2, 3, 3],
    'ProdName': ['Product A', 'Product B', 'Product A', 'Product B', 'Product A', 'Product B'],
    'TotalStockQuantity': [100, 150, 200, 80, 90, 60],
    'TotalSalesQuantity': [110, 140, 190, 70, 100, 50]
}

df = pd.DataFrame(data)

def data_ingestion():
    return df

def data_transformation(data):
    data['StockShortage'] = data['TotalStockQuantity'] - data['TotalSalesQuantity']
    return data

def model_building(data):
    X = data[['TotalStockQuantity', 'TotalSalesQuantity']]
    y = np.where(data['StockShortage'] < 0, 1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test

def recommendation_engine(model, X_test, original_data):
    predictions = model.predict(X_test)
    recommendations = []
    for i, pred in enumerate(predictions):
        status = 'Shortage' if pred > 0.5 else 'Sufficient Stock'
        recommendations.append({
            'StoreID': original_data.iloc[i]['StoreID'],
            'ProdName': original_data.iloc[i]['ProdName'],
            'Status': status
        })
    return recommendations

data = data_ingestion()
transformed_data = data_transformation(data)
model, X_test = model_building(transformed_data)
recommendations = recommendation_engine(model, X_test, transformed_data)

for rec in recommendations:
    print(f'StoreID: {rec["StoreID"]}, ProdName: {rec["ProdName"]}, Status: {rec["Status"]}')
