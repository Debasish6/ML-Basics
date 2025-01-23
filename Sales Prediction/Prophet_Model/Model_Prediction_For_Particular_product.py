from prophet.plot import plot_plotly, plot_components_plotly
import pickle
import matplotlib.pyplot as plt
import plotly.io as pio

# Load the model from the file
with open('prophet_model.pkl', 'rb') as f:
    prophet = pickle.load(f)
print("Prophet model has been loaded from 'prophet_model.pkl'.")


product_name = input("Enter the name of Product: ")

future = prophet.make_future_dataframe(periods=720)

forecast = prophet.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Plot the results 
fig1=plot_plotly(prophet, forecast)
fig2=plot_components_plotly(prophet, forecast)

pio.write_html(fig1, file='forecast_plot.html', auto_open=True)
pio.write_html(fig2, file='components_plot.html', auto_open=True)


# fig3 = prophet.plot(forecast) 
# fig3.suptitle(f'Forecast for {product_name}', fontsize=16)
# plt.show()