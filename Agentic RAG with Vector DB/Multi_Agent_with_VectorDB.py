import typer
import os
from typing import Optional,List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector,SearchType
from dotenv import load_dotenv

load_dotenv()
os.environ['Groq_API_Key'] = os.getenv('Groq_API_Key')

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector(table_name="recipes", db_url=db_url, search_type=SearchType.hybrid),
)

# Load the knowledge base: Comment out after first run
knowledge_base.load(recreate=True, upsert=True)

storage = PgAssistantStorage(table_name="pdf_assistant",db_url=db_url)

def pdf_assistant(new: bool=False, user: str = "user"):
    run_id: Optional[str] = None
    if new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user) 
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        search_knowledge=True,
        show_tool_calls=True,
        read_chat_history=True,
    )
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)


if __name__ == "__main__":
    typer.run(pdf_assistant)