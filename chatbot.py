"""
=================================================================================
CHATBOT.PY - LangGraph Chatbot with SQLite Persistence
=================================================================================

PURPOSE:
    This file creates an AI chatbot using LangGraph that can:
    - Have multi-turn conversations with memory
    - Store all conversations in SQLite database
    - Generate and store topics for each conversation thread
    - Persist chat history across app restarts

MAIN COMPONENTS:
    1. LLM Setup: HuggingFace model configuration
    2. State Definition: What data the graph stores
    3. Graph Nodes: Functions that process messages
    4. Checkpointer: SQLite database for persistence
    5. Helper Functions: Utility functions for thread management

DATABASE STRUCTURE:
    - Database File: 'chatbot.db' (SQLite)
    - Stores: 
        * All chat messages (user + AI)
        * Thread IDs (conversation identifiers)
        * Topics for each thread
        * Complete conversation state
    
FLOW:
    User Input → Graph → Chat Node → AI Response → Save to DB
                  ↓
            Checkpointer (SQLite)
                  ↓
        Stores: messages, topic, thread_id

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

# Load environment variables (API keys, etc.)
load_dotenv()


# =================================================================================
# LLM SETUP - Language Model Configuration
# =================================================================================

# HuggingFace endpoint setup
# repo_id: The model we're using from HuggingFace
# task: Type of task (conversational = chat)
llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", task="conversational")

# Wrapper to make the endpoint compatible with LangChain
model = ChatHuggingFace(llm=llm)


# =================================================================================
# STATE DEFINITION - What data does our graph store?
# =================================================================================

class ChatState(TypedDict):
    """
    The state of our chatbot graph. This is what gets stored in the database.
    
    Fields:
        messages: List of all messages in the conversation
                  - Includes both user messages (HumanMessage) and AI responses (AIMessage)
                  - add_messages: Special function that appends new messages to existing list
        
        topic: A 2-3 word summary of the conversation
               - Generated from the first user message
               - Used for displaying conversation names in sidebar
               - Stored in database for persistence
    
    Example State:
        {
            'messages': [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
                HumanMessage(content="How are you?"),
                AIMessage(content="I'm doing great!")
            ],
            'topic': "Greeting Chat"
        }
    """
    messages: Annotated[list[BaseMessage], add_messages]  # Conversation history
    topic: str  # Conversation topic/name


# =================================================================================
# GRAPH NODES - Functions that process data
# =================================================================================

def chat_node(state: ChatState):
    """
    The main processing node - handles all chat interactions.
    
    What it does:
        1. Takes the current state (with all messages)
        2. Adds a system message with bot's identity
        3. Sends everything to the AI model
        4. Returns the AI's response
    
    Args:
        state: Current ChatState with all messages
    
    Returns:
        Dictionary with new messages to add to state
        Format: {"messages": [AIMessage(content="response")]}
    
    Flow:
        Input State → Add System Message → Send to LLM → Get Response → Return Response
    """
    messages = state['messages']
    
    # System message defines the bot's personality and identity
    # This is sent with EVERY request to ensure consistent behavior
    system_message = SystemMessage(content="""You are OM, an AI assistant created by Om.""")
    
    # Combine system message with conversation history
    full_messages = [system_message] + messages
    
    # Get AI response
    response = model.invoke(full_messages)
    
    # Return in format that LangGraph expects
    return {"messages": [response]}


# =================================================================================
# DATABASE SETUP - Persistence Layer
# =================================================================================

# Create/connect to SQLite database
# check_same_thread=False: Allows multiple threads to access DB (needed for Streamlit)
connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Checkpointer: Automatically saves graph state to database after each step
# This is what makes conversations persistent across restarts
checkpointer = SqliteSaver(conn=connection)


# =================================================================================
# GRAPH CONSTRUCTION - Building the conversation flow
# =================================================================================

# Create the graph with our state definition
graph = StateGraph(ChatState)

# Add the chat node (our main processing function)
graph.add_node("chat_node", chat_node)

# Define the flow:
# START → chat_node → END
# Meaning: Every conversation starts → goes through chat_node → ends
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile the graph with checkpointer
# This creates the final bot object that we'll use in Streamlit
bot = graph.compile(checkpointer=checkpointer)


# =================================================================================
# HELPER FUNCTIONS - Utilities for managing conversations
# =================================================================================

def retrieve_all_threads():
    """
    Get list of all conversation thread IDs from database.
    
    Returns:
        list: All unique thread IDs
        Example: ['thread-123', 'thread-456', 'thread-789']
    
    How it works:
        - checkpointer.list(None) returns all checkpoints in database
        - Each checkpoint has a thread_id in its config
        - We use set() to avoid duplicates
        - Convert to list for easier use
    """
    all_threads = set()  # Set automatically removes duplicates
    
    # Loop through all checkpoints in database
    for checkpoint in checkpointer.list(None):
        # Extract thread_id from checkpoint config
        thread_id = checkpoint.config['configurable']['thread_id']
        all_threads.add(thread_id)
    
    return list(all_threads)


def generate_topic(messages):
    """
    Generate a 2-3 word topic from the first user message.
    
    Args:
        messages: The first message from user
        
    Returns:
        str: A short topic like "Weather Query" or "Python Help"
    
    Example:
        Input: "How do I use Python loops?"
        Output: "Python Loops"
    
    This is used to create readable names for conversations in the sidebar.
    """
    prompt = f"Generate a 2-3 words chat topic for following User message: {messages}"
    return model.invoke(prompt).content


def save_topic_to_thread(thread_id, topic):
    """
    Save a topic to a specific thread in the database.
    
    Args:
        thread_id: The conversation thread ID
        topic: The topic string to save
    
    What it does:
        - Uses bot.update_state() to add topic to thread's state
        - This gets saved to database automatically
        - Topic persists even after app restart
    
    Database Impact:
        Updates the 'topic' field in ChatState for this thread_id
    """
    bot.update_state(
        config={'configurable': {'thread_id': thread_id}},
        values={'topic': topic}
    )


def get_topic_from_thread(thread_id):
    """
    Retrieve a topic from a specific thread.
    
    Args:
        thread_id: The conversation thread ID
        
    Returns:
        str: The topic if found, otherwise 'New Chat'
    
    How it works:
        - Gets the full state for this thread from database
        - Extracts the 'topic' field
        - Returns 'New Chat' if no topic exists yet
    """
    try:
        # Get current state from database
        state = bot.get_state(config={'configurable': {'thread_id': thread_id}})
        
        # Extract topic, default to 'New Chat' if not found
        return state.values.get('topic', 'New Chat')
    except:
        # If thread doesn't exist or error occurs
        return 'New Chat'


def retrieve_all_threads_with_topics():
    """
    Get all threads with their topics in one go.
    
    Returns:
        dict: {thread_id: topic, ...}
        Example: {
            'thread-123': 'Weather Query',
            'thread-456': 'Python Help',
            'thread-789': 'New Chat'
        }
    
    Used by Streamlit to populate the sidebar with conversation names.
    
    Performance Note:
        - Collects unique thread IDs first (fast)
        - Then queries topic for each (one DB call per thread)
        - Results are cached in Streamlit session state
    """
    threads_with_topics = {}
    seen_threads = set()
    
    # Step 1: Collect all unique thread IDs
    for checkpoint_tuple in checkpointer.list(None):
        thread_id = checkpoint_tuple.config['configurable']['thread_id']
        seen_threads.add(thread_id)
    
    # Step 2: Get topic for each thread
    for thread_id in seen_threads:
        topic = get_topic_from_thread(thread_id)
        threads_with_topics[thread_id] = topic
    
    return threads_with_topics


# =================================================================================
# END OF FILE
# =================================================================================