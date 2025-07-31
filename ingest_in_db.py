import os
import logging
import glob
import time
from dotenv import load_dotenv
from supabase import create_client
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core import get_supabase, get_embeddings
from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("ingest.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20))
def insert_batch(supabase, data):
    """Insert a batch of documents into Supabase"""
    try:
        start_time = time.time()
        response = supabase.table("documents").insert(data).execute()
        elapsed_time = time.time() - start_time
        logger.info(f"Inserted {len(data)} chunks successfully in {elapsed_time:.2f} seconds")
        return response
    except Exception as e:
        logger.error(f"Batch insert error: {str(e)}")
        raise

def ingest_documents():
    """Ingest PDFs into Supabase documents table"""
    try:
        # Initialize Supabase and embeddings
        supabase = get_supabase()
        embeddings = get_embeddings()

        # Test Supabase connection
        response = supabase.table("documents").select("id").limit(1).execute()
        logger.info(f"Supabase connection successful: {response}")

        # Load PDFs
        pdf_dir = os.getenv("PDF_DIR", "documents")
        if not os.path.exists(pdf_dir):
            logger.error(f"PDF directory '{pdf_dir}' does not exist")
            raise FileNotFoundError(f"PDF directory '{pdf_dir}' does not exist")

        # Get list of PDF files
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in '{pdf_dir}'")
            raise ValueError(f"No PDF files found in '{pdf_dir}'")

        documents = []
        for file_path in pdf_files:
            try:
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    if len(reader.pages) == 0:
                        raise ValueError("Empty PDF file")
                    if not reader.pages[0].extract_text().strip():
                        raise ValueError("PDF appears to be image-based")
                single_loader = PyPDFLoader(file_path)
                docs = single_loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"Skipped {os.path.basename(file_path)}: {str(e)}")
                continue

        if not documents:
            logger.warning("No valid PDFs loaded")
            raise ValueError("No valid PDFs loaded")

        logger.info(f"Loaded {len(documents)} pages from PDFs in {pdf_dir}")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(splits)} chunks")

        # Prepare data for Supabase
        data = [
            {
                "content": split.page_content.replace('\x00', '').strip(),
                "metadata": {
                    "source_pdf": f"/documents/{os.path.basename(split.metadata.get('source', 'unknown.pdf'))}",
                    "page": split.metadata.get("page", 0) + 1,
                    "doc_id": f"doc_{i}",
                    "ingested_at": "2025-07-30T11:03:00Z"
                }
            }
            for i, split in enumerate(splits) if split.page_content.strip()
        ]

        if not data:
            logger.warning("No valid chunks to insert")
            raise ValueError("No valid chunks to insert")

        # Batch insert (25 chunks per batch)
        batch_size = 25
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            logger.info(f"Inserting batch {i//batch_size + 1} with {len(batch)} chunks")
            insert_batch(supabase, batch)

        logger.info(f"Inserted {len(data)} chunks into documents table")
        logger.info("Ingestion complete. Verify at https://supabase.com")

    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise

if __name__ == "__main__":
    ingest_documents()
