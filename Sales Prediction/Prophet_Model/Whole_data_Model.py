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

try:
    df = pd.read_sql(query, engine)
    engine.dispose()
except Exception as e:
    print(f"Error fetching data: {e}")
    sys.exit(1)

df['billdate'] = pd.to_datetime(df['billdate'], format='%d/%m/%Y')

wholesale_data = df.groupby('billdate')['docamountinr'].sum().reset_index()
wholesale_data.columns = ['ds', 'y']

try:
    wholesale_prophet = Prophet()
    wholesale_prophet.fit(wholesale_data)

    folder_path = r"C:\Users\eDominer\Python Project\Sales Prediction\Time Series Prediction\Prophet_Model\All_Product_model"
    model_filename = 'wholesale_prophet_model.pkl'
    full_path = os.path.join(folder_path, model_filename)

    with open(full_path, 'wb') as f:
        pickle.dump(wholesale_prophet, f)
    print("Wholesale Prophet model has been saved to 'wholesale_prophet_model.pkl'.")

except Exception as e:
    print(f"Error during Prophet modeling : {e}")
    sys.exit(1)



