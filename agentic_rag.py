import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from core import get_llm, retrieve_stream, get_supabase
from supabase import SupabaseException
from postgrest.exceptions import APIError
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("agentic_rag.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

SYSTEM_PROMPT = """You are an evidentiary assistant. Follow these steps:
1. Review all available interactions from chat_history to understand context.

2. Retrieve documents using the provided tool and list EXACT sources in this format:
[Source: filename.pdf (Page X, Year YYYY)]

3. Provide a concise, streaming answer using ONLY these sources and relevant chat_history, sentence by sentence. Compare nuances or updates across years if multiple sources apply, explaining differences. If no relevant sources are found, switch to general Amazon expertise and note: 'Using general Amazon expertise as transcripts are unavailable.'

Sources available:
{retrieved_documents}"""

def get_prompt():
    """Return the ChatPromptTemplate"""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

def extract_sources(output: str) -> list:
    """Extract sources with year from agent output"""
    if "[Source:" in output:
        sources_block, _ = output.split("\n\n", 1)
        return [s.strip() for s in sources_block.split("\n") if "[Source:" in s]
    return []

def extract_answer(output: str) -> str:
    """Extract answer from agent output"""
    if "[Source:" in output:
        _, answer = output.split("\n\n", 1)
        return answer.strip()
    return output.strip()

def extract_preview(output: str, answer: str) -> str:
    """Extract relevant preview based on answer content"""
    if "Preview:" in output:
        _, preview_block = output.split("Preview:", 1)
        previews = preview_block.strip().split("\n")
        for preview in previews:
            if any(keyword in preview.lower() for keyword in answer.lower().split()):
                return preview[:200] + "..." if len(preview) > 200 else preview
    return ""

async def query_agent_stream(question: str, user_id: str, chat_history: list) -> list:
    """Query the agent with streaming and store result"""
    try:
        supabase = get_supabase()
        chat_history_messages = chat_history
        chat_history_messages = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" 
            else AIMessage(content=f"{msg['content']}\nSources: {', '.join(msg.get('sources', []))}")
            for msg in chat_history_messages
        ]

        agent = create_tool_calling_agent(get_llm(), [retrieve_stream], get_prompt())
        executor = AgentExecutor(agent=agent, tools=[retrieve_stream], verbose=True)

        result = await executor.ainvoke({
            "input": question,
            "chat_history": chat_history_messages,
            "retrieved_documents": "",
        })

        output = result.get("output", "No response from agent")
        sources = extract_sources(output)
        answer = extract_answer(output)
        preview = extract_preview(output, answer)

        chunks = []
        for sentence in answer.split(". "):
            if sentence:
                chunks.append({"answer": sentence + ". "})
                await asyncio.sleep(0.1)  # Simulate streaming delay
        if sources:
            chunks.append({"sources": sources})
        if preview:
            chunks.append({"preview": preview})

        supabase.table("chat_history").insert({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()

        logger.info(f"Processed question: {question} for user: {user_id}")
        return chunks

    except SupabaseException as e:
        logger.error(f"Supabase error for '{question}': {str(e)}")
        return [{"answer": f"Supabase error: {str(e)}", "sources": ["Error retrieving sources"]}]
    except APIError as e:
        logger.error(f"Database error for '{question}': {str(e)}")
        return [{"answer": f"Database error: {str(e)}", "sources": ["Error retrieving sources"]}]
    except Exception as e:
        logger.error(f"Unexpected error for '{question}': {str(e)}")
        return [{"answer": f"Unexpected error: {str(e)}", "sources": ["Error retrieving sources"]}]

if __name__ == "__main__":
    question = input("Enter your question: ")
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(query_agent_stream(question, "default_user", []))
    for chunk in result:
        if "answer" in chunk:
            print(chunk["answer"])
        if "sources" in chunk:
            print("Sources:\n" + "\n".join(chunk["sources"]))
