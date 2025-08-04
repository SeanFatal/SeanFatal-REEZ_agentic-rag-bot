import streamlit as st
import logging
import os
from dotenv import load_dotenv
from supabase import create_client, Client, SupabaseException
from postgrest.exceptions import APIError
import openai
from openai import AuthenticationError
from agentic_rag import query_agent_stream

# Security Change 1: Set logging to WARNING level
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Agentic RAG",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

load_dotenv()

@st.cache_resource
def init_supabase() -> Client:
    """Initialize and cache Supabase client"""
    logger.info("Initializing Supabase client")
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

def main():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("🔑 OpenAI API key not configured!")
        st.stop()

    st.title("🔍 REEZ BOT")
    
    with st.sidebar:
        st.write("### Usage Guide")
        st.markdown("""
        1. Enter your user ID (email-style)
        2. Ask document-related questions
        3. See answers with source citations
        4. Review history below
        """)

    user_id = st.text_input("Enter your user ID:", value="default_user")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_area("Ask about your documents:", height=150, key="question_input")
    
    if st.button("Submit") and question.strip():
        with st.spinner("Analyzing..."):
            process_question(question.strip(), user_id)
    
    display_chat_history(user_id)

def process_question(question: str, user_id: str):
    placeholder = st.empty()
    full_answer = ""
    full_sources = []
    preview_content = ""

    try:
        for chunk in query_agent_stream(question, user_id, st.session_state.chat_history):
            if "answer" in chunk:
                full_answer += chunk["answer"]
            if "sources" in chunk:
                full_sources = chunk["sources"]
            if "preview" in chunk:
                preview_content = chunk["preview"]
            placeholder.markdown(f"**Question:** {question}\n**Answer:** {full_answer}\n**Sources Cited:** {', '.join(full_sources) if full_sources else 'Loading...'}\n**Preview:** {preview_content if preview_content else 'Loading...'}")
        
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": full_answer, "sources": full_sources, "preview": preview_content})
        
        with st.expander("💬 Latest Answer", expanded=True):
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Answer:**\n{full_answer}")
            st.markdown(f"**Sources Cited:**\n{', '.join(full_sources) if full_sources else 'No sources'}")
            if preview_content:
                st.markdown(f"**Preview of Evidence:**\n{preview_content}")

    except AuthenticationError as e:
        st.error("❌ Invalid OpenAI API key")
        logger.critical("OpenAI authentication failed: %s", str(e))
    except SupabaseException as e:
        st.error(f"🔌 Database error: {str(e)}")
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
            .limit(50)\
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
