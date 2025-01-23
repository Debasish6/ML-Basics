from prophet.plot import plot_plotly, plot_components_plotly
import pickle
import matplotlib.pyplot as plt
import plotly.io as pio
import re,json

# Load the model from the file
with open('wholesale_prophet_model.pkl', 'rb') as f:
    wholesale_prophet = pickle.load(f)
print("Prophet model has been loaded from 'wholesale_prophet_model.pkl'.")

with open('product_prophet_model.pkl', 'rb') as f:
    product_prophet = pickle.load(f)
print("Prophet model has been loaded from 'product_prophet_model.pkl'.")

product_name = "Blue Wood Utensil Holder"
escaped_product_name = re.escape(product_name)




# Creating future dataframes
product_forecast = product_prophet.make_future_dataframe(periods=120)
product_forecast = product_prophet.predict(product_forecast)
product_forecast1 = product_forecast[['ds', 'yhat', 'yhat_lower',  'yhat_upper']]

product_forecast_json = product_forecast1.to_json(orient='records', date_format='iso')
with open('product_forecast.json', 'w') as json_file:
    json.dump(product_forecast_json, json_file)
print("Product-based forecast saved to 'product_forecast.json'.")

wholesale_forecast = wholesale_prophet.make_future_dataframe(periods=365)
wholesale_forecast = wholesale_prophet.predict(wholesale_forecast)
wholesale_forecast1 = wholesale_forecast[['ds', 'yhat', 'yhat_lower',  'yhat_upper']]

wholesale_forecast_json = wholesale_forecast1.to_json(orient='records', date_format='iso')
with open('wholesale_forecast.json', 'w') as json_file:
    json.dump(wholesale_forecast_json, json_file)
print("Total forecast saved to 'wholesale_forecast.json'.")

product_fig = plot_plotly(product_prophet, product_forecast)
product_fig.update_layout(title=f'Forecast for {product_name} Product')
pio.write_html(product_fig, file='product_forecast.html', auto_open=True)
print("Product-based forecast saved to 'product_forecast.html'.")

wholesale_fig = plot_plotly(wholesale_prophet, wholesale_forecast)
product_fig.update_layout(title='Forecast for Product')
pio.write_html(wholesale_fig, file='wholesale_forecast.html', auto_open=True)
print("Wholesale forecast saved to 'wholesale_forecast.html'.")

import json

# Open and read the JSON file
with open('product_forecast.json', 'r') as json_file:
    data = json.load(json_file)

# Print the data
print(type(data))

