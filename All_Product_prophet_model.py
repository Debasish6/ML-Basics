import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine
import os, sys
from dotenv import load_dotenv
import re
import pickle

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

print("Database: ",db_name)

df = pd.read_sql(query, engine)
engine.dispose()

df['billdate'] = pd.to_datetime(df['billdate'], format='%d/%m/%Y')
df['unique_order'] = df['billdate'].astype(str) + '_' + df['productno'].astype(str)

df['productno'].dropna()
unique_products = df['productno'].unique()


wholesale_data = df.groupby('billdate')['docamountinr'].sum().reset_index()
wholesale_data.columns = ['ds', 'y']

wholesale_data = wholesale_data.dropna(subset=['ds', 'y'])

wholesale_prophet = Prophet()
wholesale_prophet.fit(wholesale_data)

with open(r'C:\Users\eDominer\Python Project\Sales Prediction\Time Series Prediction\Prophet_Model\All_Product_model\wholesale_prophet_model.pkl', 'wb') as f:
    pickle.dump(wholesale_prophet, f)
print("Wholesale Prophet model has been saved to 'wholesale_prophet_model.pkl'.")


for product_number in unique_products:
    if product_number != "SHIPPING R":
        escaped_product_number = re.escape(product_number)
        print(f"Processing product number: {escaped_product_number}")
        product_data = df[df['productno'].str.contains(escaped_product_number, case=False, na=False)]
        
        product_data['ds'] = pd.to_datetime(product_data['billdate'], format='%d/%m/%Y')
        product_data['y'] = product_data['docamountinr']

        product_data = product_data.dropna(subset=['ds', 'y'])

        print(f"{product_number} data after dropping NaNs: {product_data.shape}")
        print(product_data[['ds', 'y']].head())

        # Check if the product data has at least 10 rows after dropping NaNs
        if product_data.shape[0] >= 10:
            product_prophet = Prophet()
            product_prophet.fit(product_data)

            clean_product_number = re.sub(r'[\\/:*?"<>|]', '_', escaped_product_number)

            folder_path = r"C:\Users\eDominer\Python Project\Sales Prediction\Time Series Prediction\Prophet_Model\All_Product_model"
            model_filename = f'{clean_product_number}_prophet_model.pkl'
            full_path = os.path.join(folder_path, model_filename)
           
            with open(full_path, 'wb') as f:
                pickle.dump(product_prophet, f)
            print(f"{product_number}-based Prophet model has been saved to '{model_filename}'.")
        else:
            print(f"Not enough data for {product_number} to create a Prophet model. Non-NaN rows: {product_data.shape[0]}")


