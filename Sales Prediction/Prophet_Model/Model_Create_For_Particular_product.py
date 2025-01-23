import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from sqlalchemy import create_engine
import os,sys
from dotenv import load_dotenv
import re
from prophet.plot import plot_plotly, plot_components_plotly
import pickle

try:
    load_dotenv()
    db_username = os.getenv("db_username")
    db_password = os.getenv("db_password")
    db_host = os.getenv("db_hostname")
    db_name = os.getenv("db_database")
    db_server = os.getenv("db_server")
    # print(db_username, db_password, db_name, db_server)

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


# Create the Prophet model 
prophet = Prophet()

# Fit the model
prophet.fit(product_data)

# Save the model to a file
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(prophet, f)
print("Prophet model has been saved to 'prophet_model.pkl'.")

