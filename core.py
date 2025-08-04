import os
import json
import logging
from dotenv import load_dotenv
from supabase import create_client, SupabaseException
from postgrest.exceptions import APIError
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("core.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

def validate_env_vars():
    """Validate required environment variables"""
    required_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

def get_supabase():
    """Initialize Supabase client"""
    validate_env_vars()
    try:
        return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    except SupabaseException as e:
        logger.error(f"Supabase init failed: {str(e)}")
        raise RuntimeError(f"Supabase init failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during Supabase init: {str(e)}")
        raise RuntimeError(f"Unexpected error during Supabase init: {str(e)}")

def get_embeddings():
    """Initialize embeddings model"""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

def get_llm():
    """Initialize LLM"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1
    )

@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def retrieve(query: str):
    """Retrieve documents from Supabase with exact source matching and relevant preview"""
    try:
        supabase = get_supabase()
        embeddings = get_embeddings()
        vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        
        docs = vector_store.similarity_search(query, k=5)
        if not docs:
            logger.info(f"No documents found for query: {query}")
            return "No relevant documents found."

        sources = [
            f"[Source: {doc.metadata['source_pdf']} (Page {doc.metadata['page']}, Year {doc.metadata.get('year', 'Unknown')})]"
            for doc in docs
        ]
        # Extract relevant preview based on query keywords
        relevant_previews = []
        for doc in docs:
            if any(keyword in doc.page_content.lower() for keyword in query.lower().split()):
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                relevant_previews.append(f"Preview: {preview}")
        preview = "\n".join(relevant_previews) if relevant_previews else ""
        
        logger.info(f"Retrieved {len(sources)} documents for query: {query}")
        return "\n".join(sources) + "\n" + preview
    except SupabaseException as e:
        logger.error(f"Supabase error for query '{query}': {str(e)}")
        return f"Error retrieving documents: {str(e)}"
    except APIError as e:
        logger.error(f"Database error for query '{query}': {str(e)}")
        return f"Error retrieving documents: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected retrieval error for query '{query}': {str(e)}")
        return f"Error retrieving documents: {str(e)}"
Additional Dependencies
