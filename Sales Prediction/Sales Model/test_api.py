import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "Month": [1, 2, 3],
    "Year": [2025, 2025, 2025],
    "Sales_Lag_1": [3090, 698, 8195],
    "Sales_Lag_2": [3200, 780, 8300],
    "Sales_Lag_3": [3000, 700, 8100],
    "QUANTITYORDERED": [100, 150, 200],
    "PRICEEACH": [20.5, 21.0, 19.5],
    "MSRP": [25, 26, 24],
    "DEALSIZE_Small": [0, 1, 0],
    "DEALSIZE_Medium": [1, 0, 0],
    "DEALSIZE_Large": [0, 0, 1]
}


response = requests.post(url, json=data)
print(response.json())
