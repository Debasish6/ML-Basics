from phi.agent import Agent
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground,serve_playground_app
import openai,os,phi.api
from dotenv import load_dotenv

load_dotenv()

phi.api=os.getenv("Phidata_API_Key")

#Creating Web Search Agent
web_search_agent = Agent(
    name = 'Web Search Agent',
    role= 'Search the web for Information',
    model = Groq(id="llama-3.1-8b-Instant"),
    tools=[DuckDuckGo()],
    instructions=["Always Include Sources"],
    show_tool_calls=True,
    markdown=True
)

#Creating Financial Agent
financial_agent = Agent(
    name = 'Financial Agent',
    model = Groq(id="llama-3.1-8b-Instant"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

app = Playground(agents=[financial_agent,web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)