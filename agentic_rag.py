import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from core import get_llm, retrieve, get_supabase
from supabase import SupabaseException
from postgrest.exceptions import APIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("agentic_rag.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

# System prompt with chat history and strict sourcing
SYSTEM_PROMPT = """You are an evidentiary assistant. Follow these steps:
1. Review the last 5 interactions from chat history to understand context.

2. List EXACT sources from retrieved documents in this format:
[Source: filename.pdf (Page X)]
[Source: report.docx (Page Y)]

3. Provide a concise answer using ONLY these sources and relevant chat history.

4. If no relevant sources or chat history, state: 'Based on available documents and chat history, I cannot provide a definitive answer.'

Sources available:
{retrieved_documents}

User question: {input}"""

def get_prompt():
    """Return the ChatPromptTemplate"""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

def extract_sources(output: str) -> list:
    """Extract sources from agent output as a list"""
    if "[Source:" in output:
        sources_block, _ = output.split("\n\n", 1)
        return [s.strip() for s in sources_block.split("\n") if "[Source:" in s]
    return ["No sources cited"]

def extract_answer(output: str) -> str:
    """Extract answer from agent output"""
    if "[Source:" in output:
        _, answer = output.split("\n\n", 1)
        return answer.strip()
    return output.strip()

def create_agent():
    """Create agent with source-enforcement and chat history"""
    llm = get_llm()
    prompt = get_prompt()
    agent = create_tool_calling_agent(llm, [retrieve], prompt)
    return AgentExecutor(agent=agent, tools=[retrieve], verbose=True)

def query_agent(question: str, user_id: str = "default_user") -> dict:
    """Query the agent and store result in chat history"""
    try:
        # Retrieve chat history
        supabase = get_supabase()
        chat_history = supabase.table("chat_history")\
            .select("question, answer, sources")\
            .eq("user_id", user_id)\
            .order("timestamp", desc=True)\
            .limit(5)\
            .execute()
        
        # Convert chat history to list of BaseMessage objects
        chat_history_messages = []
        if chat_history.data:
            for entry in chat_history.data:
                chat_history_messages.append(HumanMessage(content=entry['question']))
                chat_history_messages.append(AIMessage(content=f"{entry['answer']}\nSources: {', '.join(entry['sources'])}"))
        else:
            chat_history_messages = []

        # Query agent
        agent = create_agent()
        result = agent.invoke({
            "input": question,
            "chat_history": chat_history_messages,
            "retrieved_documents": "",  # Will be populated by retrieve tool
        })
        
        # Parse output
        output = result.get("output", "No response from agent")
        sources = extract_sources(output)
        answer = extract_answer(output)
        
        # Store in chat history
        supabase.table("chat_history").insert({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "sources": sources,  # Store as list directly
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
        
        logger.info(f"Processed question: {question} for user: {user_id}")
        return {"sources": sources, "answer": answer}

    except SupabaseException as e:
        logger.error(f"Supabase client error for '{question}': {str(e)}")
        return {"sources": ["Error retrieving sources"], "answer": f"Supabase error: {str(e)}"}
    except APIError as e:
        logger.error(f"Database query error for '{question}': {str(e)}")
        return {"sources": ["Error retrieving sources"], "answer": f"Database error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error for '{question}': {str(e)}")
        return {"sources": ["Error retrieving sources"], "answer": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    question = input("Enter your question: ")
    result = query_agent(question)
    print(f"Sources:\n{'\n'.join(result['sources'])}\n\nAnswer:\n{result['answer']}")
