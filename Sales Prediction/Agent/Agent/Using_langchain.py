from langchain_groq import ChatGroq  # Use the chat-based model
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from langchain_core.runnables.base import RunnableSequence
import openai,pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("Groq_API_Key")

llm = ChatGroq(model="llama3-8b-8192",api_key=api_key, temperature=0.1)


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
Ensure that give small answer.
"""

prompt = PromptTemplate(input_variables=["data"], template=prompt_template)

stock_data = {
    "StoreID": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    "Product": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"] * 6,
    "Stock": [100, 90, 80, 70, 60, 50, 150, 140, 130, 120, 110, 100, 200, 180, 160, 140, 130, 120, 80, 70, 60, 55, 50, 45, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35],
    "Sales": [90, 85, 80, 80, 75, 70, 140, 130, 120, 110, 100, 90, 190, 175, 150, 135, 125, 115, 70, 60, 55, 50, 45, 40, 100, 95, 90, 85, 80, 75, 50, 45, 40, 35, 30, 25]
}

df = pd.DataFrame(stock_data)

sequence = RunnableSequence(prompt, llm)

prediction = sequence.invoke({"data": stock_data})

print(prediction.content)

import markdown

html = markdown.markdown(prediction.content)

with open('stock_vs_sales.html', 'w') as file:
    file.write(html)

print("HTML content saved to stock_vs_sales.html")

