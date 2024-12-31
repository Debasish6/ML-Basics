import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import markdown

# Load environment variables
load_dotenv()

def clear_console():
    """Clear the console for both Windows and Unix-like systems."""
    if sys.platform.startswith('win'):
        os.system('cls')
    else:
        os.system('clear')

# Setting up GPT-4o mini Model, Embedding models, and Vector Store
model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0, model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

# Loading PDF
loader = PyMuPDFLoader(r"C:\Users\eDominer\Python Project\ChatBot\ChatBot_with_Database\OpenAI Tuned Model\Costing Detail Screen documentation.docx.pdf")
docs = loader.load()

# Splitting Documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

# Indexing Chunks
document_ids = vector_store.add_documents(documents=all_splits)

# Graph builder
graph_builder = StateGraph(MessagesState)

# Creating Tools
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    model_with_tools = model.bind_tools([retrieve])
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Step 2: Execute the retrieval
tools = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content
def generate(state: MessagesState):
    """Generate answer."""
    recent_tool_messages = [message for message in reversed(state["messages"]) if message.type == "tool"]
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    system_message_content = (
        "You are an AI assistant for eDominer Technologies Pvt Ltd. Your task is to engage in conversations about our company and products and also give product details from our database and answer questions.Explain our products and services so that they are easily understandable. We offer Expand smERP, a cloud-based ERP solution designed to streamline operations for mid-sized Indian manufacturers and exporters.\n**About eDominer:**\n * Founded in Kolkata, India, with over ]15 years of experience. \n* Led by a team of experts in technology and business automation.\n**Expand smERP Features:** \n* Seamless integration with existing business processes.\n* Automation of complex tasks for increased efficiency.\n* User-friendly interface with minimal training required.\n* Secure data storage on Microsoft Azure with SSL encryption.\n* Integration with popular platforms like WhatsApp, Paytm, and Amazon.\n* Customizable options to fit specific business needs.\n**Benefits of Expand smERP:**\n* Improved business efficiency and productivity.\n* Reduced costs through automation and streamlined processes.\n* Enhanced data security and management.\n* Scalable solution to grow with your business.        **Our Plans**\n1. Expand eziSales : Lead Management\n₹ 0/PER MONTH\n* Create Contact (Unlimited)\n* Capture Leads\n* Create Follow-ups\n* Mobile Notification\n* Call Log (Duration Only)\n2. Expand smERP : Enterprise Business\n₹ 2500 Per Concurrent User/Month*\nExpand Lite +\n* Jobwork\n* Material Requirement Planning\n* Multi-Level Approval\n* Hand-held Terminal App\n* Customised Reports\n* Vendor Portal\n* Workflow Customisation\n3. Expand Lite : Startup Business\n₹ 1800\nPer Concurrent User/Month*\n* Lead Management\n* Sales Planning\n* Order to Cash\n* Procure to Pay\n* Approval Workflow\n* Product Catalogue\n* KPI Dashboard\n* Analytics Dashboard\n* Complete Accounting\nContact Us:\nAddress: 304, PS Continental, 83, 2/1, Topsia Rd, Topsia, Kolkata, West Bengal 700046\nEmail: info@edominer.com\nPhone: +91 9007026542\nProduct Website: https://www.expanderp.com/aboutus/\nWebsite : https://www.edominer.com/\n**Ask me anything about eDominer or Expand smERP!"
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = model.invoke(prompt)
    return {"messages": [response]}

# Graph Configuration
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123", "checkpoint_ns": "some_namespace", "checkpoint_id": "some_checkpoint_id"}}

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


while True:
    user_input = input().strip()
    if user_input.lower() == "bye":
        print("Goodbye!")
        break
    elif user_input.lower() == "train":
        print("Training started...")
        sys.stdout.flush()
        # Simulate a training process or replace with real training logic
        vector_store.add_documents([])  # Placeholder for training logic
        print("Training complete!")
        sys.stdout.flush()
    else:
        try:
            for step in graph.stream({"messages": [{"role": "user", "content": user_input}]}, stream_mode="values", config=config):
                clear_console()
                filtered_messages = [msg for msg in step["messages"] if msg.type != "tool"]
                for message in filtered_messages[1:]:
                    
                    if message.content.strip() != '':
                        html = markdown.markdown(message.content)
                        print(html,"\n---Expand AI Return---")
        except Exception as e:
            print(f"Error: {e}")
            sys.stdout.flush()