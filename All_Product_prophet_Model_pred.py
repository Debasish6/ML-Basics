import pandas as pd
import pickle
import os
from pathlib import Path
import plotly.graph_objs as go
import plotly.io as pio
from prophet.plot import plot_plotly
import json

class SalesForecaster:
    def __init__(self, model_dir, html_folder_path, json_folder_path):
        self.model_dir = Path(model_dir)
        self.html_folder_path = Path(html_folder_path)
        self.json_folder_path = Path(json_folder_path)
        self.figs = []

    def load_model(self, model_file):
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading model {model_file}: {e}")
            return None

    def save_forecast(self, product_name, forecast):
        forecast1 = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast_json = forecast1.to_json(orient='records', date_format='iso')
        json_filename = self.json_folder_path / f"{product_name}_forecast.json"
        with open(json_filename, 'w') as json_file:
            json.dump(forecast_json, json_file)
        print(f"Forecast saved to '{json_filename}'.")

    def save_plot(self, product_name, fig):
        html_filename = self.html_folder_path / f"{product_name}_product_forecast.html"
        pio.write_html(fig, file=html_filename, auto_open=False)
        print(f"Product-based forecast saved to '{html_filename}'.")

    def forecast_product(self, model, product_name):
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        fig = plot_plotly(model, forecast)
        fig.update_layout(title=f'Forecast for {product_name} Product')
        self.save_plot(product_name, fig)
        self.save_forecast(product_name, forecast)
        self.figs.append(fig)

    def run(self):
        model_files = [f for f in self.model_dir.iterdir() if f.name.endswith('_prophet_model.pkl')]
        for model_file in model_files:
            product_name = model_file.stem.split('_prophet_model')[0]
            print("product_name:",product_name)
            model = self.load_model(model_file)
            if model:
                self.forecast_product(model, product_name)



model_dir = r"C:\Users\eDominer\Python Project\Sales Prediction\Time Series Prediction\Prophet_Model\All_Product_model"
html_folder_path = r"C:\Users\eDominer\Python Project\Sales Prediction\Time Series Prediction\Prophet_Model\All_Product_HTML"
json_folder_path = r"C:\Users\eDominer\Python Project\Sales Prediction\Time Series Prediction\Prophet_Model\All_Product_JSON"

forecaster = SalesForecaster(model_dir, html_folder_path, json_folder_path)
forecaster.run()

