import os
import sys
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
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
from IPython.display import Image, display

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Set up your models, embeddings, and vector store
model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0, model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

# Loading the document for chatbot's knowledge base
loader = PyMuPDFLoader(r"C:\Users\eDominer\Python Project\ChatBot\ChatBot_with_Database\OpenAI Tuned Model\Help_whole.pdf")
docs = loader.load()

# Splitting Documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

# Indexing the chunks
document_ids = vector_store.add_documents(documents=all_splits)

# Graph builder setup for chat flow
graph_builder = StateGraph(MessagesState)

# Create tools and responses
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    model_with_tools = model.bind_tools([retrieve])
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools = ToolNode([retrieve])

def generate(state: MessagesState):
    """Generate answer based on the latest state and tool calls."""
    recent_tool_messages = [message for message in reversed(state["messages"]) if message.type == "tool"]
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)

    system_message_content = (
        "You are an AI assistant. Ask me anything about our products and services!"
        f"{docs_content}"
    )
    conversation_messages = [
        message for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = model.invoke(prompt)
    return {"messages": [response]}

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

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Configuring the required keys for the checkpointer
config = {
    "configurable": {
        "thread_id": "abc123",  # Unique ID for the conversation thread
        "checkpoint_ns": "some_namespace",  # Namespace for the checkpoint (can be arbitrary)
        "checkpoint_id": "some_checkpoint_id"  # Unique checkpoint ID for saving state
    }
}


# Route for the homepage (simple interface)
@app.route('/')
def home():
    return render_template('index.html')  # Create a simple HTML page to interact with

# API endpoint for chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    if user_message:
        # Generate the response from the chatbot
        for step in graph.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            stream_mode="values",
            config=config,
        ):
            filtered_messages = [msg for msg in step["messages"] if msg.type != "tool"]
            response_message = ""
            for message in filtered_messages:
                if message.content.strip():
                    response_message += message.content + "\n"
            
            return jsonify({"response": response_message.strip()})
    return jsonify({"response": "Error: No message received"})

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app locally
