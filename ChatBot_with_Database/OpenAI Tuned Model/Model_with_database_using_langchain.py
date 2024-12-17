from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Dummy Callbacks class for demonstration purposes
class Callbacks:
    pass

# Define a minimal BaseCache model (for demonstration)
class BaseCache(BaseModel):
    some_field: str = Field(default="default_value")

# Rebuild the SQLDatabaseChain model
SQLDatabaseChain.model_rebuild()

def connect_to_database():
    """Loads database credentials from a dotenv file and creates a connection engine."""
    db_username = os.getenv("db_username")
    db_password = os.getenv("db_password")
    db_host = os.getenv("db_hostname")
    db_name = os.getenv("db_database")
    db_server = os.getenv("db_server")
    connection_string = f"mssql+pyodbc://{db_username}:{db_password}@{db_server}/{db_name}?driver=ODBC Driver 17 for SQL Server"
    engine = create_engine(connection_string)
    return engine

def main():
    """Connects to the database, initializes OpenAI client, and runs the query."""

    # Connect to database
    engine = connect_to_database()

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=api_key, temperature=0.3)

    # Create the SQL database chain
    db = SQLDatabase(engine=engine)
    #print(db.get_table_info())
    chain = SQLDatabaseChain.from_llm(llm,db, verbose=True)

    # Define the query
    query = "How many products are there in products table?"

    # Run the query and print the result
    result = chain.invoke(query)
    print(result)

if __name__ == "__main__":
    main()
