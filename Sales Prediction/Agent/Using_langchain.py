from langchain_groq import ChatGroq # Use the chat-based model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("Groq_API_Key")

llm = ChatGroq(model="llama3-8b-8192")  # Use ChatOpenAI for chat models

prompt_template = """
Given the following stock data, predict the stock needs for the next month for each product:

{data}

The prediction should consider sales trends, product popularity, and stock shortages.

Provide predictions in the following format:

- StoreID: 1, Product: Product A, Predicted Stock: <predicted_value>
- StoreID: 2, Product: Product B, Predicted Stock: <predicted_value>
"""

prompt = PromptTemplate(input_variables=["data"], template=prompt_template)

stock_data = """
Store 1 - Product A: Stock = 100, Sales = 110
Store 1 - Product B: Stock = 150, Sales = 140
Store 2 - Product A: Stock = 200, Sales = 190
Store 2 - Product B: Stock = 80, Sales = 70
Store 3 - Product A: Stock = 90, Sales = 100
Store 3 - Product B: Stock = 60, Sales = 50
"""

llm_chain = LLMChain(llm=llm, prompt=prompt)

prediction = llm_chain.run({"data": stock_data})

print(prediction)
