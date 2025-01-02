from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai,os
from dotenv import load_dotenv

load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")


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


Multi_AI_agent =Agent(
    team=[web_search_agent,financial_agent],
    instructions=["Always include sources","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

Multi_AI_agent.print_response("Summerize analyst recommendation and share the latest news for Nvidia",stream=True)