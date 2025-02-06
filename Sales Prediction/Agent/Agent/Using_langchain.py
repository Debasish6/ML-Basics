from langchain_groq import ChatGroq  # Use the chat-based model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("Groq_API_Key")

llm = ChatGroq(model="llama3-8b-8192",api_key=api_key, temperature=0.3)  # Use ChatOpenAI for chat models


prompt_template = """
Given the following stock data for the past six months, predict the stock needs for the next month for each product:

{data}

The prediction should consider sales trends, product popularity, and stock shortages.

Additionally, provide recommendations to move products from stores with lower demand to stores with higher demand.

Provide predictions and recommendations in the following format:

- StoreID: 1, Product: Product A, Predicted Stock: <predicted_value>
- StoreID: 2, Product: Product B, Predicted Stock: <predicted_value>

Recommendations:
- Move <quantity> of Product A from Store <low_demand_store_id> to Store <high_demand_store_id>
"""

prompt = PromptTemplate(input_variables=["data"], template=prompt_template)

# Sample stock data for six months (date-wise)
stock_data = """
Store 1 - Product A:
- Jan: Stock = 100, Sales = 110
- Feb: Stock = 90, Sales = 95
- Mar: Stock = 80, Sales = 85
- Apr: Stock = 70, Sales = 80
- May: Stock = 60, Sales = 75
- Jun: Stock = 50, Sales = 70

Store 1 - Product B:
- Jan: Stock = 150, Sales = 140
- Feb: Stock = 140, Sales = 130
- Mar: Stock = 130, Sales = 120
- Apr: Stock = 120, Sales = 110
- May: Stock = 110, Sales = 100
- Jun: Stock = 100, Sales = 90

Store 2 - Product A:
- Jan: Stock = 200, Sales = 190
- Feb: Stock = 180, Sales = 175
- Mar: Stock = 160, Sales = 150
- Apr: Stock = 140, Sales = 135
- May: Stock = 130, Sales = 125
- Jun: Stock = 120, Sales = 115

Store 2 - Product B:
- Jan: Stock = 80, Sales = 70
- Feb: Stock = 70, Sales = 60
- Mar: Stock = 60, Sales = 55
- Apr: Stock = 55, Sales = 50
- May: Stock = 50, Sales = 45
- Jun: Stock = 45, Sales = 40

Store 3 - Product A:
- Jan: Stock = 90, Sales = 100
- Feb: Stock = 85, Sales = 95
- Mar: Stock = 80, Sales = 90
- Apr: Stock = 75, Sales = 85
- May: Stock = 70, Sales = 80
- Jun: Stock = 65, Sales = 75

Store 3 - Product B:
- Jan: Stock = 60, Sales = 50
- Feb: Stock = 55, Sales = 45
- Mar: Stock = 50, Sales = 40
- Apr: Stock = 45, Sales = 35
- May: Stock = 40, Sales = 30
- Jun: Stock = 35, Sales = 25
"""

llm_chain = LLMChain(llm=llm, prompt=prompt)

prediction = llm_chain.run({"data": stock_data})

print(prediction)
