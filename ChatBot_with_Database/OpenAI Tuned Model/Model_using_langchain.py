# from openai import OpenAI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()
# client = OpenAI(api_key =os.getenv("OpenAIKey"))

# print("We are good to go...")
#prompt = PromptTemplate.from_template("What is the capital of {place}")

llm = ChatOpenAI(api_key =os.getenv("OPENAI_API_KEY"),temperature=0.3)

# chain = prompt | llm 

# city ="Delhi"
# output = chain.invoke(city)


# print(output)

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.expanderp.com/aboutus")

docs = loader.load()

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

from langchain.chains.combine_documents import create_stuff_documents_chain

prompt = PromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# from langchain_core.documents import Document

# document_chain.invoke({
#     "input": "how can langsmith help with testing?",
#     "context": [Document(page_content="langsmith can let you visualize test results")]
# })

from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "Why we choose Expand smERP?"})
print(response["answer"])

# LangSmith offers several features that can help with testing:...