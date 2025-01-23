import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from sqlalchemy import create_engine
import os, sys
from dotenv import load_dotenv
import re
import pickle
import plotly.graph_objs as go
from prophet.plot import plot_plotly
import plotly.io as pio

try:
    load_dotenv()
    db_username = os.getenv("db_username")
    db_password = os.getenv("db_password")
    db_host = os.getenv("db_hostname")
    db_name = os.getenv("db_database")
    db_server = os.getenv("db_server")

    connection_string = f'mssql+pyodbc://{db_username}:{db_password}@{db_server}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server'
    engine = create_engine(connection_string)
except Exception as e:
    print("Error connecting to SQL Server:", e)
    sys.exit(1)

query = os.getenv("SQL_QUERY")

df = pd.read_sql(query, engine)
engine.dispose()

df['billdate'] = pd.to_datetime(df['billdate'], format='%d/%m/%Y')
df['unique_order'] = df['billdate'].astype(str) + '_' + df['productno'].astype(str)

product_name = input("Enter the name of Product: ")
escaped_product_name = re.escape(product_name)
print(escaped_product_name)

product_data = df[df['prodname'].str.contains(escaped_product_name, case=False, na=False)]
print(product_data[['unique_order', 'docamount', 'docstatusname']])

product_data['ds'] = pd.to_datetime(product_data['billdate'], format='%d/%m/%Y')
product_data['y'] = product_data['docamountinr']

product_prophet = Prophet()
product_prophet.fit(product_data)

with open('product_prophet_model.pkl', 'wb') as f:
    pickle.dump(product_prophet, f)
print("Product-based Prophet model has been saved to 'product_prophet_model.pkl'.")

wholesale_data = df.groupby('billdate')['docamountinr'].sum().reset_index()
wholesale_data.columns = ['ds', 'y']


wholesale_prophet = Prophet()
wholesale_prophet.fit(wholesale_data)

with open('wholesale_prophet_model.pkl', 'wb') as f:
    pickle.dump(wholesale_prophet, f)
print("Wholesale Prophet model has been saved to 'wholesale_prophet_model.pkl'.")


