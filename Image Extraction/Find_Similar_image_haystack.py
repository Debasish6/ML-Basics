from haystack import Pipeline
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes.retriever.multimodal import MultiModalRetriever
import os

class MultimodalImageSearch:
    def __init__(self, doc_dir):
        self.document_store = InMemoryDocumentStore(embedding_dim=512)
        self.doc_dir = doc_dir

        images = [
            Document(content=filename, content_type="image")
            for filename in os.listdir(doc_dir)
        ]

        self.document_store.write_documents(images)

        self.retriever_text_to_image = MultiModalRetriever(
            document_store=self.document_store,
            query_embedding_model="sentence-transformers/clip-ViT-B-32",
            query_type="text",
            document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},
        )

        self.pipeline = Pipeline()
        self.pipeline.add_node(component=self.retriever_text_to_image, name="retriever", inputs=["Query"])

    def search(self, query, top_k=3):
        # Update embeddings before searching for accurate results
        self.document_store.update_embeddings(retriever=self.retriever_text_to_image)
        results = self.pipeline.run(query=query, params={"retriever": {"top_k": top_k}})

        # Extract distances for relevance score
        return sorted(
            results["documents"], key=lambda d: d.meta["distance"], reverse=True
        )[:top_k]


# searcher = MultimodalImageSearch(doc_dir="C:/MKExim/KB/Archives/Products")
# results = searcher.search(query="Find images of laptops")
# for result in results:
#     print(f"Image: {result.content}, Distance (Relevance): {result.meta['distance']}")