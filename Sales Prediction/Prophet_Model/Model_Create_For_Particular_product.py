import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine
import os, sys
from dotenv import load_dotenv
import re
import pickle
import platform

def clear_console():
    command = 'cls' if platform.system().lower() == 'windows' else 'clear'
    os.system(command)

try:
    load_dotenv()
    db_username = os.getenv("db_username")
    db_password = os.getenv("db_password")
    db_host = os.getenv("db_hostname")
    db_name = os.getenv("db_database")
    db_server = os.getenv("db_server")

    connection_string = f'mssql+pyodbc://{db_username}:{db_password}@{db_server}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server'
    engine = create_engine(connection_string)
except KeyError as ke:
    print(f"Missing environment variable: {ke}")
    sys.exit(1)
except Exception as e:
    print("Error connecting to SQL Server:", e)
    sys.exit(1)

try:
    query = os.getenv("SQL_QUERY")
    df = pd.read_sql(query, engine)
    engine.dispose()
except ValueError as ve:
    print(f"Error in SQL query: {ve}")
except Exception as e:
    print(f"General error fetching data: {e}")

try:
    df['billdate'] = pd.to_datetime(df['billdate'], format='%d/%m/%Y')
    df['unique_order'] = df['billdate'].astype(str) + '_' + df['productno'].astype(str)
    df['productno'].dropna()
    unique_products = df['productno'].unique()

    lower_limit = int(input("Enter the lower limit of the data:"))
    upper_limit = int(input("Enter the upper limit of the data:"))
    length = len(unique_products)

    upper_limit = upper_limit if upper_limit <= length else length

    print(f"Product from {lower_limit} to {upper_limit}: ", unique_products[lower_limit:upper_limit])

    for product_number in unique_products[lower_limit:upper_limit]:
        if product_number != "SHIPPING R":
            escaped_product_number = re.escape(product_number)
            #print(f"Processing product number: {escaped_product_number}")
            product_data = df[df['productno'].str.contains(escaped_product_number, case=False, na=False)]
            
            product_data['ds'] = pd.to_datetime(product_data['billdate'], format='%d/%m/%Y')
            product_data['y'] = product_data['docamountinr']
            product_data = product_data.dropna(subset=['ds', 'y'])

            if product_data.shape[0] >= 10:
                product_prophet = Prophet()
                product_prophet.fit(product_data)

                clean_product_number = re.sub(r'[\\/:*?"<>|]', '_', escaped_product_number)
                folder_path = r"C:\Users\eDominer\Python Project\Sales Prediction\Time Series Prediction\Prophet_Model\All_Product_model"
                model_filename = f'{clean_product_number}_prophet_model.pkl'
                full_path = os.path.join(folder_path, model_filename)
            
                with open(full_path, 'wb') as f:
                    pickle.dump(product_prophet, f)
                # print(f"{product_number}-based Prophet model has been saved to '{model_filename}'.")
            # else:
            #     print(f"Not enough data for {product_number} to create a Prophet model. Non-NaN rows: {product_data.shape[0]}")
            clear_console()
    else:
        if upper_limit<length :
            print("---Partial Tranning Successfully Done---")
        elif upper_limit == length:
            print("---Tranning Successfully Done---")

except ValueError as ve:
    print(f"Value error: {ve}")
except KeyError as ke:
    print(f"Key error: {ke}")
except Exception as e:
    print(f"Error: {e}")


