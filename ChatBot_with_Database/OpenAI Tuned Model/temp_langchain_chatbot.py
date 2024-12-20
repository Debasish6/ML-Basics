# import os,sys
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage
# from langchain_core.messages import AIMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import START, MessagesState, StateGraph
# from langchain_core.messages import SystemMessage, trim_messages
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain import hub
# from typing_extensions import List, TypedDict
# from langchain_core.documents import Document


# load_dotenv()


# from langchain_openai import ChatOpenAI

# model = ChatOpenAI(api_key =os.getenv("OPENAI_API_KEY"),temperature=0,model="gpt-4o-mini")

# print("Connected Model Succesfully")


# model.invoke(
#     [
#         HumanMessage(content="Hi! I'm Deba"),
#         AIMessage(content="Hello Deba! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ]
# )

# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are an AI assistant for eDominer Technologies Pvt Ltd. Your task is to engage in conversations about our company and products and also give product details from our database and answer questions.Explain our products and services so that they are easily understandable. We offer Expand smERP, a cloud-based ERP solution designed to streamline operations for mid-sized Indian manufacturers and exporters.\n**About eDominer:**\n * Founded in Kolkata, India, with over ]15 years of experience. \n* Led by a team of experts in technology and business automation.\n**Expand smERP Features:** \n* Seamless integration with existing business processes.\n* Automation of complex tasks for increased efficiency.\n* User-friendly interface with minimal training required.\n* Secure data storage on Microsoft Azure with SSL encryption.\n* Integration with popular platforms like WhatsApp, Paytm, and Amazon.\n* Customizable options to fit specific business needs.\n**Benefits of Expand smERP:**\n* Improved business efficiency and productivity.\n* Reduced costs through automation and streamlined processes.\n* Enhanced data security and management.\n* Scalable solution to grow with your business.        **Our Plans**\n1. Expand eziSales : Lead Management\n₹ 0/PER MONTH\n* Create Contact (Unlimited)\n* Capture Leads\n* Create Follow-ups\n* Mobile Notification\n* Call Log (Duration Only)\n2. Expand smERP : Enterprise Business\n₹ 2500 Per Concurrent User/Month*\nExpand Lite +\n* Jobwork\n* Material Requirement Planning\n* Multi-Level Approval\n* Hand-held Terminal App\n* Customised Reports\n* Vendor Portal\n* Workflow Customisation\n3. Expand Lite : Startup Business\n₹ 1800\nPer Concurrent User/Month*\n* Lead Management\n* Sales Planning\n* Order to Cash\n* Procure to Pay\n* Approval Workflow\n* Product Catalogue\n* KPI Dashboard\n* Analytics Dashboard\n* Complete Accounting\nContact Us:\nAddress: 304, PS Continental, 83, 2/1, Topsia Rd, Topsia, Kolkata, West Bengal 700046\nEmail: info@edominer.com\nPhone: +91 9007026542\nProduct Website: https://www.expanderp.com/aboutus/\nWebsite : https://www.edominer.com/\n**Ask me anything about eDominer or Expand smERP!**\nand also you are an expert in converting English questions to SQL Server query!\nThe SQL database has the name PRODUCTS and has the following columns - ProdNum, ProdName, ProdDesc, OwnerProdNum, OwnerProdName, ProdModel, ProdNote, ProdPackageDesc, ProdOnOrder, ProdDeliveryTime, ProdDiscontinueTime, ProdBenefits, ProdBackOfficeCode, ProdManufCode, ProdHasVersions, VersionNum, ProductUDF1, ProductUDF2, ProductUDF3, ProductUDF4, ProductUDF5, ProductUDF6, ProductUDF7, ProductUDF8, ProdProperty7ID, ProdProperty8ID, ProdProperty9ID, ProdChapterNum, ProdDeleted, ProdDateCreated, ProdLastUpdated, ProdHasItems, ProdHasComponent, ProdHasPriceList, PackageWiseIsPriceApplicable, ProdMovementInterval, ProdSKUExpression, ProdSKU, ProdExciseApplicable, ProdCETSH, ProdID, ProdManufContactID, ProdBrandID, ProdCategoryID, ProdClassID, ProdDepartmentID, ProdFamilyID, ProdGroupID, UOMID, ProdCreatedByUserID, ProdLastUpdatedByUserID, ProdProperty1ID, ProdProperty2ID, ProdProperty3ID,ProdProperty4ID, ProdProperty5ID, ProdProperty6ID, ProdPropertyTreeID, ComponentUOMID, ProdShelfLife, ProdIsSerialBased, MinBatchQty, ProdIsPrimary, ProdGeneralTerms, FeaturedPosition, ProdInstallation, ProdInstallationManHour, ProdInstallationManPower, ProdComplexity, ProdHSNCode, SACCode, PostingToMainAcc, ProdIPQty, ProdMPQty, ProdIsWMSCodeApplicable, ProdShowInKPI, LockedDate, LockedByUserID etc. Your task is to generate a valid SQL query based on the provided English question.\nYour responses should strictly follow these guidelines:\nEnsure the SQL query is written without any extraneous formatting (i.e., no markdown, no backticks, no SQL keyword).\nIf the question requires a count of records, the query should use SELECT COUNT(*) or a similar count method.\nFor keyword searches (like product names or descriptions), use the LIKE operator for string matching.\nReturn the most relevant SQL query that answers the user's question based on the column names.\nFor example,\nExample 1 - How many entries of records are present?, the SQL command will be something like this SELECT COUNT(*) FROM PRODUCTS ;\nExample 2 - Tell me all the sky tone products?, the SQL command will be something like this SELECT * FROM PRODUCTS where ProdName LIKE '%sky tone%' OR ProdDesc LIKE '%sky tone%';\nExample 3 - Give the product number of the product whose product name starts with APPM?, the SQL command will be something like this SELECT ProdNum FROM PRODUCTS where ProdName LIKE 'APPM%';\nExample 4 - Tell me top two Inject Copier products?, the SQL command will be something like this SELECT TOP 2 ProdNum, ProdName FROM PRODUCTS WHERE ProdName LIKE '%Inject Copier%' ORDER BY ProdName;\nExample 5 - Tell me the Product Name whose Product back office code is 4COPI047A, the SQL command will be something like this SELECT ProdName FROM PRODUCTS WHERE ProdBackOfficeCode = '4COPI047A';\nExample 6 - What is the product name for the product with ProdNum PRO/0278, the SQL command will be something like this SELECT ProdName FROM PRODUCTS WHERE ProdNum = 'PRO/0278';\nExample 7 - Show me all the products created in the year 2023., the SQL command will be something like this SELECT * FROM PRODUCTS WHERE YEAR(ProdDateCreated) = 2023;\nExample 8 - Give me the hsn code of sky tone, the SQL command will be something like this SELECT Prod Name, ProdHSNCode FROM PRODUCTS WHERE ProdName LIKE '%sky tone%' OR ProdDesc LIKE '%sky tone%';\nExample 9 - List the product descriptions for products that have the word 'blue' in their name., the SQL command will be something like this SELECT ProdDesc FROM PRODUCTS WHERE ProdName LIKE '%blue%'; and you can add multiple columns also in sql query for accurate result.\nExample 10 - Tell me which product has highest entries., the SQL command will be something like this SELECT Top 1 ProdName, COUNT(*) AS EntryCount FROM PRODUCTS GROUP BY ProdName ORDER BY EntryCount DESC;\nExample 11 - Tell me which product has second highest entries., the SQL command will be something like this WITH RankedProducts AS (SELECT ProdName, COUNT(*) AS EntryCount,ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS RowNum FROM PRODUCTS GROUP BY ProdName) SELECT ProdName, EntryCount FROM RankedProducts WHERE RowNum = 2;\nalso the sql code should not have ``` in the beginning or end and sql word in output",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )


# trimmer = trim_messages(
#     max_tokens=65,
#     strategy="last",
#     token_counter=model,
#     include_system=True,
#     allow_partial=False,
#     start_on="human",
# )

# messages = [
#     SystemMessage(content="you're a good assistant"),
#     HumanMessage(content="hi! I'm bob"),
#     AIMessage(content="hi!"),
#     HumanMessage(content="I like vanilla ice cream"),
#     AIMessage(content="nice"),
#     HumanMessage(content="whats 2 + 2"),
#     AIMessage(content="4"),
#     HumanMessage(content="thanks"),
#     AIMessage(content="no problem!"),
#     HumanMessage(content="having fun?"),
#     AIMessage(content="yes!"),
# ]

# trimmer.invoke(messages)

# #Loading PDF
# loader = PyMuPDFLoader(r"C:\Users\eDominer\Python Project\ChatBot\ChatBot_with_Database\OpenAI Tuned Model\Help_whole.pdf")

# docs = loader.load()
# #print(docs[0].page_content)


# #Splitting Documents
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,  # chunk size (characters)
#     chunk_overlap=200,  # chunk overlap (characters)
#     add_start_index=True,  # track index in original document
# )
# all_splits = text_splitter.split_documents(docs)


# #Loading data into vector
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# from langchain_core.vectorstores import InMemoryVectorStore
# # Initialize with an embedding model
# vector_store = InMemoryVectorStore(embeddings)

# document_ids = vector_store.add_documents(documents=all_splits)


# # Retrieval and Generation
# # Define prompt for question-answering
# prompt = hub.pull("rlm/rag-prompt")

# # Define state for application
# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str


# def generate(state: State):
#     # Combine the retrieved documents into a single string
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
#     # Prepare the messages in the format expected by the prompt_template
#     messages = [
#         {"role": "system", "content": "You are an AI assistant for eDominer Technologies..."},
#         {"role": "user", "content": state["question"]},
#         {"role": "assistant", "content": docs_content}
#     ]
    
#     input_data = {"messages": messages}
    
#     prompt_response = prompt_template.invoke(input=input_data)
    
#     response = model.invoke(prompt_response)
    
#     return {"answer": response.content}


# def retrieve(state: State):
#     # Retrieve relevant documents based on the question
#     retrieved_docs = vector_store.similarity_search(state["question"])
#     return {"context": retrieved_docs}

# # Set up state and graph
# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")

# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)

# # Get user input and generate response
# config = {"configurable": {"thread_id": "abc123"}}

# while True:
#     user_input = input("You: ")
#     if user_input!='bye':
#             response = graph.invoke({"question": user_input}, config)
#             print(response["answer"])
#     else:
#         print('Good Bye!')
#         break



import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

# Setting up GPT-4o mini Model, Embedding models, and Vector Store
model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0, model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

# Loading PDF
loader = PyMuPDFLoader(r"C:\Users\eDominer\Python Project\ChatBot\ChatBot_with_Database\OpenAI Tuned Model\Help_whole.pdf")
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
    
    # Define the ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant for eDominer Technologies Pvt Ltd. Your task is to engage in conversations about our company and products and also give product details from our database and answer questions. Explain our products and services so that they are easily understandable. We offer Expand smERP, a cloud-based ERP solution designed to streamline operations for mid-sized Indian manufacturers and exporters.\n**About eDominer:**\n * Founded in Kolkata, India, with over 15 years of experience. \n* Led by a team of experts in technology and business automation.\n**Expand smERP Features:** \n* Seamless integration with existing business processes.\n* Automation of complex tasks for increased efficiency.\n* User-friendly interface with minimal training required.\n* Secure data storage on Microsoft Azure with SSL encryption.\n* Integration with popular platforms like WhatsApp, Paytm, and Amazon.\n* Customizable options to fit specific business needs.\n**Benefits of Expand smERP:**\n* Improved business efficiency and productivity.\n* Reduced costs through automation and streamlined processes.\n* Enhanced data security and management.\n* Scalable solution to grow with your business.        **Our Plans**\n1. Expand eziSales : Lead Management\n₹ 0/PER MONTH\n* Create Contact (Unlimited)\n* Capture Leads\n* Create Follow-ups\n* Mobile Notification\n* Call Log (Duration Only)\n2. Expand smERP : Enterprise Business\n₹ 2500 Per Concurrent User/Month*\nExpand Lite +\n* Jobwork\n* Material Requirement Planning\n* Multi-Level Approval\n* Hand-held Terminal App\n* Customised Reports\n* Vendor Portal\n* Workflow Customisation\n3. Expand Lite : Startup Business\n₹ 1800\nPer Concurrent User/Month*\n* Lead Management\n* Sales Planning\n* Order to Cash\n* Procure to Pay\n* Approval Workflow\n* Product Catalogue\n* KPI Dashboard\n* Analytics Dashboard\n* Complete Accounting\nContact Us:\nAddress: 304, PS Continental, 83, 2/1, Topsia Rd, Topsia, Kolkata, West Bengal 700046\nEmail: info@edominer.com\nPhone: +91 9007026542\nProduct Website: https://www.expanderp.com/aboutus/\nWebsite : https://www.edominer.com/\n**Ask me anything about eDominer or Expand smERP!**\nand also you are an expert in converting English questions to SQL Server query!\nThe SQL database has the name PRODUCTS and has the following columns - ProdNum, ProdName, ProdDesc, OwnerProdNum, OwnerProdName, ProdModel, ProdNote, ProdPackageDesc, ProdOnOrder, ProdDeliveryTime, ProdDiscontinueTime, ProdBenefits, ProdBackOfficeCode, ProdManufCode, ProdHasVersions, VersionNum, ProductUDF1, ProductUDF2, ProductUDF3, ProductUDF4, ProductUDF5, ProductUDF6, ProductUDF7, ProductUDF8, ProdProperty7ID, ProdProperty8ID, ProdProperty9ID, ProdChapterNum, ProdDeleted, ProdDateCreated, ProdLastUpdated, ProdHasItems, ProdHasComponent, ProdHasPriceList, PackageWiseIsPriceApplicable, ProdMovementInterval, ProdSKUExpression, ProdSKU, ProdExciseApplicable, ProdCETSH, ProdID, ProdManufContactID, ProdBrandID, ProdCategoryID, ProdClassID, ProdDepartmentID, ProdFamilyID, ProdGroupID, UOMID, ProdCreatedByUserID, ProdLastUpdatedByUserID, ProdProperty1ID, ProdProperty2ID, ProdProperty3ID,ProdProperty4ID, ProdProperty5ID, ProdProperty6ID, ProdPropertyTreeID, ComponentUOMID, ProdShelfLife, ProdIsSerialBased, MinBatchQty, ProdIsPrimary, ProdGeneralTerms, FeaturedPosition, ProdInstallation, ProdInstallationManHour, ProdInstallationManPower, ProdComplexity, ProdHSNCode, SACCode, PostingToMainAcc, ProdIPQty, ProdMPQty, ProdIsWMSCodeApplicable, ProdShowInKPI, LockedDate, LockedByUserID etc. Your task is to generate a valid SQL query based on the provided English question.\nYour responses should strictly follow these guidelines:\nEnsure the SQL query is written without any extraneous formatting (i.e., no markdown, no backticks, no SQL keyword).\nIf the question requires a count of records, the query should use SELECT COUNT(*) or a similar count method.\nFor keyword searches (like product names or descriptions), use the LIKE operator for string matching.\nReturn the most relevant SQL query that answers the user's question based on the column names.\nFor example,\nExample 1 - How many entries of records are present?, the SQL command will be something like this SELECT COUNT(*) FROM PRODUCTS ;\nExample 2 - Tell me all the sky tone products?, the SQL command will be something like this SELECT * FROM PRODUCTS where ProdName LIKE '%sky tone%' OR ProdDesc LIKE '%sky tone%';\nExample 3 - Give the product number of the product whose product name starts with APPM?, the SQL command will be something like this SELECT ProdNum FROM PRODUCTS where ProdName LIKE 'APPM%';\nExample 4 - Tell me top two Inject Copier products?, the SQL command will be something like this SELECT TOP 2 ProdNum, ProdName FROM PRODUCTS WHERE ProdName LIKE '%Inject Copier%' ORDER BY ProdName;\nExample 5 - Tell me the Product Name whose Product back office code is 4COPI047A, the SQL command will be something like this SELECT ProdName FROM PRODUCTS WHERE ProdBackOfficeCode = '4COPI047A';\nExample 6 - What is the product name for the product with ProdNum PRO/0278, the SQL command will be something like this SELECT ProdName FROM PRODUCTS WHERE ProdNum = 'PRO/0278';\nExample 7 - Show me all the products created in the year 2023., the SQL command will be something like this SELECT * FROM PRODUCTS WHERE YEAR(ProdDateCreated) = 2023;\nExample 8 - Give me the hsn code of sky tone, the SQL command will be something like this SELECT Prod Name, ProdHSNCode FROM PRODUCTS WHERE ProdName LIKE '%sky tone%' OR ProdDesc LIKE '%sky tone%';\nExample 9 - List the product descriptions for products that have the word 'blue' in their name., the SQL command will be something like this SELECT ProdDesc FROM PRODUCTS WHERE ProdName LIKE '%blue%'; and you can add multiple columns also in sql query for accurate result.\nExample 10 - Tell me which product has highest entries., the SQL command will be something like this SELECT Top 1 ProdName, COUNT(*) AS EntryCount FROM PRODUCTS GROUP BY ProdName ORDER BY EntryCount DESC;\nExample 11 - Tell me which product has second highest entries., the SQL command will be something like this WITH RankedProducts AS (SELECT ProdName, COUNT(*) AS EntryCount,ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS RowNum FROM PRODUCTS GROUP BY ProdName) SELECT ProdName, EntryCount FROM RankedProducts WHERE RowNum = 2;\n"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    # Generate response using the model with the prompt template
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
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
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

# Display Graph
display(Image(graph.get_graph().draw_mermaid_png()))

# Interactive Loop
while True:
    input_message = input("You: ")
    if input_message != 'bye':
        for step in graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
        ):
            step["messages"][-1].pretty_print()
    else:
        print('Good Bye!')
        break
