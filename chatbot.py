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
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun , WikipediaQueryRun
from langchain.utilities   import WikipediaAPIWrapper
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3
import os

# =================================================================================
# SECURE API KEY LOADING
# =================================================================================

# Load environment variables from .env file (for local development)
load_dotenv()


# =================================================================================
# LLM SETUP - Language Model Configuration (SECURE)
# =================================================================================

# Setup HuggingFace endpoint with explicit token


main_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

search_tool = DuckDuckGoSearchRun(region="us-en")
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search_tool,wikipedia_tool]
model = main_model.bind_tools(tools)


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
    
    system_message = SystemMessage(content="""You are Panda, an AI assistant created by Om. Om is a 12th Science student from India.!!!! return name of tool that you use at the last ok response.""")
    
    full_messages = [system_message] + messages
    response = model.invoke(full_messages)
    
    return {"messages": [response]}

tool_node = ToolNode(tools)

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
graph.add_node("tools",tool_node)


graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

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
