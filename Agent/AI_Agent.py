from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv

import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyBiTt_0SEhUhfrERSxzWMVsfdSw5LKzpxA"

load_dotenv()

async def main():
    agent = Agent(
       task="""Go to google and search for Ind vs Ban match and find who pick more wicket""",
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
    )
    result = await agent.run()
    print(result)

asyncio.run(main())