import streamlit as st
import logging
import os
from dotenv import load_dotenv
from agentic_rag import query_agent
from supabase import create_client, Client, SupabaseException
from postgrest.exceptions import APIError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Agentic RAG", layout="wide")

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
pooler_url = "postgres://postgres:Reezmedia12345@db.npkqpsxhllobpwdzxmcc.supabase.co:6543/postgres"  # Hardcoded for now; update in .env if needed
logger.info(f"Supabase URL: {supabase_url}")
logger.info(f"Supabase Key: {supabase_key}")
logger.info(f"Pooler URL: {pooler_url}")
supabase = create_client(supabase_url, supabase_key)

def get_supabase() -> Client:
    return supabase

def main():
    st.title("🔍 REEZ Document-Based Q&A BOT")
    
    # Input for user ID
    user_id = st.text_input("Enter your user ID (e.g., email):", value="default_user")
    
    # Question input
    with st.container():
        question = st.text_area("Ask about your documents:", height=150)
    
    if st.button("Submit") and question:
        process_question(question, user_id)
    
    # Display chat history
    display_chat_history(user_id)

def process_question(question: str, user_id: str):
    with st.spinner("Analyzing documents..."):
        try:
            result = query_agent(question, user_id)
            
            # Display latest result
            with st.expander("💬 Latest Answer", expanded=True):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:**\n{result['answer']}")
                st.markdown(f"**Sources Cited:**\n{result['sources']}")
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

def display_chat_history(user_id: str):
    try:
        supabase = get_supabase()
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
                    st.markdown(f"**Sources:**\n{', '.join(entry['sources'])}")
    except SupabaseException as e:
        st.error(f"Supabase client error loading chat history: {str(e)}")
    except APIError as e:
        st.error(f"Database query error loading chat history: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error loading chat history: {str(e)}")

if __name__ == "__main__":
    main()
