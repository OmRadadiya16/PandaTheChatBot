"""
=================================================================================
CHATBOT.PY - LangGraph Chatbot with SQLite Persistence (SECURE VERSION)
=================================================================================
API Key Security: Uses environment variables and Streamlit secrets
=================================================================================
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings

import sqlite3
import os

# =================================================================================
# SECURE API KEY LOADING
# =================================================================================

# Load environment variables from .env file (for local development)
load_dotenv()

def get_api_key():
    """
    Securely load API key from multiple sources.
    Priority: Streamlit Secrets → Environment Variables
    
    Returns:
        str: The API key
        
    Raises:
        ValueError: If no API key is found
    """
    api_key = None
    
    # Try Streamlit secrets first (for deployed apps)
    try:
        import streamlit as st
        # Check if running in Streamlit context
        if hasattr(st, 'secrets'):
            api_key = st.secrets.get("HUGGINGFACE_API_KEY", None)
            if api_key:
                print("✅ API key loaded from Streamlit secrets")
                return api_key
    except Exception as e:
        print(f"⚠️  Could not load from Streamlit secrets: {e}")
    
    # Try environment variable (for local development)
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if api_key:
        print("✅ API key loaded from environment variable")
        return api_key
    
    # If nothing found, raise error with helpful message
    raise ValueError(
        "❌ HuggingFace API key not found!\n"
        "Please set it in one of these ways:\n"
        "1. Streamlit Cloud: Settings → Secrets → Add HUGGINGFACE_API_KEY\n"
        "2. Local: Create .env file with: HUGGINGFACE_API_KEY=your_key\n"
        "3. Environment: export HUGGINGFACE_API_KEY=your_key\n"
    )

# Get API key securely
HF_API_KEY = get_api_key()


# =================================================================================
# LLM SETUP - Language Model Configuration (SECURE)
# =================================================================================

# Get API key securely
try:
    HF_API_KEY = get_api_key()
    print(f"✅ API Key loaded successfully (length: {len(HF_API_KEY)})")
except Exception as e:
    print(f"❌ Error loading API key: {e}")
    raise

# Setup HuggingFace endpoint with explicit token
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b", 
    task="conversational",
    huggingfacehub_api_token=HF_API_KEY,  # Explicit token
    # Alternative parameter names (try if above doesn't work)
    # token=HF_API_KEY,
    # api_key=HF_API_KEY,
)

model = ChatHuggingFace(llm=llm)


# =================================================================================
# STATE DEFINITION
# =================================================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    topic: str


# =================================================================================
# GRAPH NODES
# =================================================================================

def chat_node(state: ChatState):
    messages = state['messages']
    
    system_message = SystemMessage(content="""You are Panda, an AI assistant created by Om. Om is a 12th Science student from India.!!!!""")
    
    full_messages = [system_message] + messages
    response = model.invoke(full_messages)
    
    return {"messages": [response]}


# =================================================================================
# DATABASE SETUP
# =================================================================================

connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)


# =================================================================================
# GRAPH CONSTRUCTION
# =================================================================================

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

bot = graph.compile(checkpointer=checkpointer)


# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        all_threads.add(thread_id)
    return list(all_threads)


def generate_topic(messages):
    prompt = f"Generate a 2-3 words chat topic for following User message: {messages}"
    return model.invoke(prompt).content


def save_topic_to_thread(thread_id, topic):
    bot.update_state(
        config={'configurable': {'thread_id': thread_id}},
        values={'topic': topic}
    )


def get_topic_from_thread(thread_id):
    try:
        state = bot.get_state(config={'configurable': {'thread_id': thread_id}})
        return state.values.get('topic', 'New Chat')
    except:
        return 'New Chat'


def retrieve_all_threads_with_topics():
    threads_with_topics = {}
    seen_threads = set()
    
    for checkpoint_tuple in checkpointer.list(None):
        thread_id = checkpoint_tuple.config['configurable']['thread_id']
        seen_threads.add(thread_id)
    
    for thread_id in seen_threads:
        topic = get_topic_from_thread(thread_id)
        threads_with_topics[thread_id] = topic
    
    return threads_with_topics


# =================================================================================
# END OF FILE
# =================================================================================
