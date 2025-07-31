pythonimport streamlit as st
import logging
import os
from dotenv import load_dotenv
from supabase import create_client, Client, SupabaseException
from postgrest.exceptions import APIError
import openai
from openai import AuthenticationError
from agentic_rag import query_agent  # Import the updated query_agent

# Security Change 1: Set logging to WARNING level to prevent sensitive info leaks
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Structure Change 1: Page config after imports to prevent Streamlit warnings
st.set_page_config(
    page_title="Agentic RAG",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Critical Change 1: Early environment loading before any other operations
load_dotenv()

# Performance Change 1: Cached client initialization
@st.cache_resource
def init_supabase() -> Client:
    """Initialize and cache Supabase client to prevent multiple connections"""
    logger.info("Initializing Supabase client")
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )

def main():
    # Security Change 2: Mandatory OpenAI key check
    if not os.getenv("OPENAI_API_KEY"):
        st.error("🔑 OpenAI API key not configured! Please set it in Render environment variables.")
        st.stop()

    st.title("🔍 REEZ Document-Based Q&A BOT")
    
    # UX Change 1: Added sidebar guidance
    with st.sidebar:
        st.write("### Usage Guide")
        st.markdown("""
        1. Enter your user ID (email-style)
        2. Ask document-related questions
        3. Review history below
        """)

    user_id = st.text_input("Enter your user ID:", value="default_user")
    
    with st.container():
        question = st.text_area("Ask about your documents:", height=150)
    
    # Validation Change 1: Trim whitespace and empty input handling
    if st.button("Submit") and question.strip():
        process_question(question.strip(), user_id)
    
    display_chat_history(user_id)

def process_question(question: str, user_id: str):
    with st.spinner("Analyzing documents..."):
        try:
            result = query_agent(question, user_id)  # Call to agentic_rag.py
            
            with st.expander("💬 Latest Answer", expanded=True):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:**\n{result['answer']}")
                st.markdown(f"**Sources Cited:**\n{', '.join(result['sources']) if result['sources'] else 'No sources'}")
                
        except AuthenticationError as e:
            # Security Change 3: Specific OpenAI auth error handling
            st.error("❌ Invalid OpenAI API key - contact administrator or update in Render")
            logger.critical("OpenAI authentication failed: %s", str(e))
            
        except SupabaseException as e:
            st.error(f"🔌 Database connection error: {str(e)}")
            
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {str(e)}")
            logger.exception("Processing failed: %s", str(e))

def display_chat_history(user_id: str):
    try:
        supabase = init_supabase()
        
        history = supabase.table("chat_history")\
            .select("question, answer, sources, timestamp")\
            .eq("user_id", user_id)\
            .order("timestamp", desc=True)\
            .limit(10)\
            .execute()
        
        if history.data:
            st.divider()
            st.subheader("📜 Recent Chat History")
            for entry in history.data:
                with st.expander(f"Q: {entry['question']} ({entry['timestamp']})"):
                    st.markdown(f"**Answer:**\n{entry['answer']}")
                    st.markdown(f"**Sources:**\n{', '.join(entry['sources']) if entry['sources'] else 'No sources'}")
                    
    except APIError as e:
        st.error(f"📦 Database query error: {str(e)}")
        
    except Exception as e:
        st.error("🚨 Failed to load history")
        logger.error("History load error: %s", str(e))

if __name__ == "__main__":
    main()
