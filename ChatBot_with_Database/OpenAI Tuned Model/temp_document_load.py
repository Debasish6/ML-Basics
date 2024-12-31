# from langchain_community.document_loaders import PyMuPDFLoader

# loader = PyMuPDFLoader(r"C:\Users\eDominer\Python Project\ChatBot\ChatBot_with_Database\OpenAI Tuned Model\Help_whole.pdf")

# docs = loader.load()
# # print(docs[168])

# print(docs[168].page_content)
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore
import numpy as np
import sqlite3
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ast  # To safely evaluate the string back to a dictionary

# SQLiteVectorStore Implementation
class SQLiteVectorStore(VectorStore):
    def __init__(self, db_path: str, embedding_model: OpenAIEmbeddings):
        """Initialize SQLiteVectorStore."""
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

        # Initialize the database schema
        self._create_table()

    def _create_table(self):
        """Creates a table to store vectors and metadata."""
        self.cursor.execute(""" 
            CREATE TABLE IF NOT EXISTS vectors (
                doc_id TEXT PRIMARY KEY,
                vector BLOB,
                metadata TEXT
            )
        """)
        self.connection.commit()

    def _vector_to_blob(self, vector: np.ndarray) -> bytes:
        """Convert a numpy array to a BLOB format."""
        return vector.tobytes()

    def _blob_to_vector(self, blob: bytes) -> np.ndarray:
        """Convert a BLOB back into a numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        try:
            for doc in documents:
                # Ensure that 'id' is in the metadata
                if 'id' not in doc.metadata:
                    doc.metadata['id'] = str(hash(doc.page_content))  # or use another method to generate a unique ID

                # Get the embedding for each document
                embedding = self.embedding_model.embed_documents([doc.page_content])[0]
                if isinstance(embedding, list):  # If it's still a list, convert it to a numpy array
                    embedding = np.array(embedding)

                vector_blob = self._vector_to_blob(embedding)

                # Insert the document into the SQLite table
                self.cursor.execute("""
                    INSERT OR REPLACE INTO vectors (doc_id, vector, metadata) 
                    VALUES (?, ?, ?)
                """, (doc.metadata["id"], vector_blob, str(doc.metadata)))
            
            self.connection.commit()
        except Exception as e:
            print(f"Error adding documents: {e}")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve the top k most similar documents based on cosine similarity."""
        try:
            query_embedding = self.embedding_model.embed_query(query)  # Use embed_query
            if isinstance(query_embedding, list):  # If it's a list, convert it to numpy array
                query_embedding = np.array(query_embedding)
            
            print(f"Query embedding shape: {query_embedding.shape}")

            # Fetch all vectors from the database
            self.cursor.execute("SELECT doc_id, vector, metadata FROM vectors")
            rows = self.cursor.fetchall()

            similarities = []
            for row in rows:
                doc_id, vector_blob, metadata_str = row
                
                # Convert metadata string back to dictionary
                metadata = ast.literal_eval(metadata_str)  # Safely evaluate string to dictionary
                
                stored_vector = self._blob_to_vector(vector_blob)

                print(f"Stored vector shape: {stored_vector.shape}")

                if query_embedding.shape != stored_vector.shape:
                    print(f"Shape mismatch: query({query_embedding.shape}), stored({stored_vector.shape})")
                    continue  # Skip this iteration if shapes don't match

                similarity = self._cosine_similarity(query_embedding, stored_vector)
                similarities.append((doc_id, similarity, metadata))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k_results = similarities[:k]

            return [Document(page_content="", metadata=metadata) for _, _, metadata in top_k_results]

        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2)

    def close(self):
        """Close the database connection."""
        self.connection.close()

    def from_texts(self, texts: List[str], metadata: List[dict] = None) -> List[Document]:
        """
        Convert a list of texts into a list of Document objects. 
        Each document can optionally have metadata.
        """
        if metadata is None:
            metadata = [{"id": str(i)} for i in range(len(texts))]  # Default metadata with unique IDs

        documents = []
        for text, meta in zip(texts, metadata):
            # Ensure 'id' is in the metadata
            if "id" not in meta:
                meta["id"] = str(hash(text))  # Generate a unique ID using the hash of the text

            documents.append(Document(page_content=text, metadata=meta))

        return documents


# Example Usage
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Print the loaded OpenAI API Key (for debugging purposes)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("OpenAI API Key loaded successfully.")
    else:
        print("Error: OpenAI API Key not loaded.")

    # Initialize the embedding model
    embedding_model = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")

    # Create SQLite vector store
    db_path = r"C:\Users\eDominer\Python Project\ChatBot\ChatBot_with_Database\OpenAI Tuned Model\vectors.db"
    vector_store = SQLiteVectorStore(db_path=db_path, embedding_model=embedding_model)

    # Create sample documents by loading a PDF
    loader = PyMuPDFLoader(r"C:\Users\eDominer\Python Project\ChatBot\ChatBot_with_Database\OpenAI Tuned Model\Customer Preference.pdf")
    docs = loader.load()

    # Splitting Documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    # Add documents to the vector store
    print("Adding documents to vector store...")
    vector_store.add_documents(documents=all_splits)

    # Perform a similarity search
    query = "What is Customer preference?"
    print(f"\nPerforming similarity search for query: {query}")
    results = vector_store.similarity_search(query, k=2)

    # Display the results
    for result in results:
        print(f"Found document with metadata: {result.metadata}")

    # Close the connection when done
    vector_store.close()
