from prophet.plot import plot_plotly, plot_components_plotly
import pickle
import matplotlib.pyplot as plt
import plotly.io as pio
import re,json


try:
    # Load the model from the file
    model_path = r"C:\Users\eDominer\Python Project\Sales Prediction\Time Series Prediction\Prophet_Model\All_Product_model\wholesale_prophet_model.pkl"
    html_folder_path = r"C:\Users\eDominer\Python Project\Sales Prediction\Time Series Prediction\Prophet_Model\All_Product_HTML\wholesale_forecast.html"
    json_folder_path = r"C:\Users\eDominer\Python Project\Sales Prediction\Time Series Prediction\Prophet_Model\All_Product_JSON\wholesale_forecast.json"

    with open(model_path, 'rb') as f:
        wholesale_prophet = pickle.load(f)
    print("Prophet model has been loaded from 'wholesale_prophet_model.pkl'.")

    wholesale_forecast = wholesale_prophet.make_future_dataframe(periods=365)
    wholesale_forecast = wholesale_prophet.predict(wholesale_forecast)
    wholesale_forecast1 = wholesale_forecast[['ds', 'yhat', 'yhat_lower',  'yhat_upper']]

    wholesale_forecast_json = wholesale_forecast1.to_json(orient='records', date_format='iso')
    # print(wholesale_forecast_json)
    with open(json_folder_path, 'w') as json_file:
        json.dump(wholesale_forecast_json, json_file)
    print("Total forecast saved to 'wholesale_forecast.json'.")

    wholesale_fig = plot_plotly(wholesale_prophet, wholesale_forecast)
    wholesale_fig.update_layout(title='Forecast for Product')
    pio.write_html(wholesale_fig, file=html_folder_path, auto_open=False)
    print("Wholesale forecast saved to 'wholesale_forecast.html'.")
except FileNotFoundError as e:
    print(f"Error loading model or data: {e}")
except pickle.PickleError as e:
    print(f"Error with pickle operation: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

try:
    with open('product_forecast.json', 'r') as json_file:
        data = json.load(json_file)
    print("Product forecast data loaded from 'product_forecast.json'.")
except FileNotFoundError as e:
    print(f"Error loading JSON file: {e}")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")



