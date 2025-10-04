"""
=================================================================================
STREAMLIT APP - LangGraph Chatbot UI
=================================================================================

PURPOSE:
    This is the user interface (UI) for the chatbot. It provides:
    - Chat interface for talking to the bot
    - Sidebar with list of all conversations
    - Ability to create new chats
    - Ability to switch between conversations
    - Automatic topic generation and display

MAIN SECTIONS:
    1. Imports & Setup
    2. Utility Functions
    3. Session State Management
    4. Sidebar UI (conversation list)
    5. Main Chat UI

SESSION STATE VARIABLES:
    st.session_state stores data that persists across Streamlit reruns:
    
    - 'thread_id': Current conversation ID (string like 'abc-123-def')
    - 'message_history': Current chat messages for display
        Format: [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi!'}]
    - 'chat_threads': All threads with topics
        Format: {'thread-123': 'Weather Query', 'thread-456': 'Python Help'}
    - 'needs_thread_reload': Flag to reload threads from database

UI FLOW:
    1. User opens app → Load threads from DB → Show sidebar
    2. User types message → Generate topic (if first msg) → Send to bot → Display response
    3. User clicks sidebar button → Load that conversation → Display messages
    4. User clicks "New Chat" → Create new thread → Clear display

=================================================================================
"""

import streamlit as st
from chatbot import (
    bot,                              # The compiled LangGraph chatbot
    retrieve_all_threads,             # Get all thread IDs
    generate_topic,                   # Generate topic from message
    save_topic_to_thread,             # Save topic to database
    get_topic_from_thread,            # Get topic from database
    retrieve_all_threads_with_topics  # Get all threads with their topics
)
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# =================================================================================
# UTILITY FUNCTIONS - Helper functions for app functionality
# =================================================================================

def generate_thread_id():
    """
    Generate a unique thread ID for a new conversation.
    
    Returns:
        str: A unique ID like 'abc-123-def-456'
    
    Why UUID:
        - Guaranteed to be unique
        - No collisions even with millions of threads
        - Works across multiple users/sessions
    """
    thread_id = uuid.uuid4()
    return str(thread_id)


def reset_chat():
    """
    Create a brand new chat conversation.
    
    What it does:
        1. Generates a new unique thread_id
        2. Adds it to session state
        3. Clears message history (fresh start)
    
    Called when:
        - User clicks "New Chat" button
        - User first opens the app
    
    Effect:
        - Sidebar shows new "New Chat" entry
        - Main chat area is empty
        - Ready for first message
    """
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def add_thread(thread_id):
    """
    Add a thread to the session state list.
    
    Args:
        thread_id: The thread ID to add
    
    Why needed:
        - New threads don't exist in database yet (no messages)
        - But we want them to appear in sidebar immediately
        - So we add them to session state with 'New Chat' label
    
    Effect:
        Ensures thread appears in sidebar even before first message
    """
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_id] = 'New Chat'


def load_conversation(thread_id):
    """
    Load all messages from a specific conversation thread.
    
    Args:
        thread_id: The conversation to load
        
    Returns:
        list: All messages (HumanMessage and AIMessage objects)
        
    Example Return:
        [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm great!")
        ]
    
    How it works:
        - Calls bot.get_state() to get full state from database
        - Extracts just the 'messages' field
        - Returns empty list if thread has no messages yet
    """
    state = bot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


# =================================================================================
# SESSION STATE SETUP - Initialize variables on first run
# =================================================================================
# Note: This section runs every time Streamlit reruns, but the 'if not in' checks
#       ensure initialization only happens once per session

# Initialize message history for current conversation
# Stores: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Initialize current thread ID
# This is the conversation we're currently viewing/using
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# Load all threads with their topics from database
# Format: {'thread-123': 'Weather Query', 'thread-456': 'Python Help'}
# Only loads ONCE per session for performance
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads_with_topics()

# Flag for tracking when to reload threads
# Used to refresh sidebar after operations
if 'needs_thread_reload' not in st.session_state:
    st.session_state['needs_thread_reload'] = False

# Ensure current thread is in the list
# Important: New threads won't be in database yet, but should appear in sidebar
add_thread(st.session_state['thread_id'])


# =================================================================================
# SIDEBAR UI - Conversation List
# =================================================================================

st.sidebar.title('LangGraph Chatbot')

# "New Chat" button - creates fresh conversation
if st.sidebar.button('New Chat'):
    reset_chat()
    st.session_state['needs_thread_reload'] = True
    st.rerun()  # Refresh page to show new chat

st.sidebar.header('My Conversations')

# Check if there are any threads to display
if not st.session_state['chat_threads']:
    st.sidebar.info("No conversations yet. Start chatting!")
else:
    # Get all threads and reverse (newest first)
    # Example: [('thread-789', 'New Chat'), ('thread-456', 'Python Help'), ...]
    thread_items = list(st.session_state['chat_threads'].items())[::-1]

    # Create a button for each conversation
    for thread_id, topic in thread_items:
        # Button shows topic name (not ugly thread ID)
        # key=f"btn_{thread_id}" ensures unique button IDs (Streamlit requirement)
        if st.sidebar.button(topic, key=f"btn_{thread_id}"):
            # User clicked this conversation - load it!
            
            # Update current thread ID
            st.session_state['thread_id'] = thread_id
            
            # Load all messages from database
            messages = load_conversation(thread_id)

            # Convert LangChain messages to Streamlit format
            temp_messages = []
            for msg in messages:
                # Determine role based on message type
                role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})

            # Update session state with loaded messages
            st.session_state['message_history'] = temp_messages
            
            # Refresh page to display loaded conversation
            st.rerun()


# =================================================================================
# MAIN CHAT UI - The actual chat interface
# =================================================================================

# Display all messages in current conversation
# This runs every time page reloads, showing the full chat history
for message in st.session_state['message_history']:
    # st.chat_message creates a chat bubble (user on right, assistant on left)
    with st.chat_message(message['role']):
        st.markdown(message['content'])  # Display message text

# Chat input box at bottom
# Returns None if user hasn't typed anything
user_input = st.chat_input('Type here')

# If user submitted a message
if user_input:
    # =============================================================================
    # TOPIC GENERATION - Only for first message in new thread
    # =============================================================================
    
    # Check if this thread already has a topic
    current_topic = st.session_state['chat_threads'].get(st.session_state['thread_id'], 'New Chat')
    
    # If it's still "New Chat", this is the first message - generate topic
    if current_topic == 'New Chat':
        with st.spinner('Generating topic...'):  # Show loading spinner
            # Generate topic from user's message
            topic = generate_topic(user_input)
            
            # Save topic to database
            save_topic_to_thread(st.session_state['thread_id'], topic)
            
            # Update session state so sidebar shows new topic immediately
            st.session_state['chat_threads'][st.session_state['thread_id']] = topic

    # =============================================================================
    # DISPLAY USER MESSAGE
    # =============================================================================
    
    # Add to message history for display
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    
    # Show user message in chat
    with st.chat_message('user'):
        st.markdown(user_input)

    # =============================================================================
    # GET AI RESPONSE
    # =============================================================================
    
    # Config tells LangGraph which thread to use (for loading/saving state)
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # Create AI message area
    with st.chat_message("assistant"):
        # Stream AI response token by token (for better UX)
        def ai_only_stream():
            """
            Generator function that yields AI response chunks.
            
            How streaming works:
                1. bot.stream() processes message and returns chunks
                2. We filter to only AIMessage chunks (ignore metadata)
                3. Yield each chunk's content
                4. st.write_stream() displays them in real-time
            
            Why stream:
                - Better user experience (see response as it generates)
                - Feels more interactive
                - Shows bot is "thinking"
            """
            for message_chunk, metadata in bot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                # Only yield AI messages (filter out other message types)
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        # Display streamed response and get final message
        ai_message = st.write_stream(ai_only_stream())

    # =============================================================================
    # SAVE AI RESPONSE TO HISTORY
    # =============================================================================
    
    # Add AI response to message history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    
    # Rerun to refresh sidebar (in case topic was just generated)
    st.rerun()


# =================================================================================
# END OF FILE
# =================================================================================


# =================================================================================
# SUMMARY - How everything works together:
# =================================================================================

# 1. APP STARTS:
#    - Load all threads from database
#    - Create/load current thread_id
#    - Display sidebar with conversation list

# 2. USER TYPES MESSAGE:
#    - Check if first message → generate & save topic
#    - Display user message
#    - Send to bot → get AI response
#    - Display AI response
#    - Save everything to database (automatic via checkpointer)

# 3. USER CLICKS CONVERSATION:
#    - Load messages from database
#    - Update display
#    - Switch thread_id

# 4. USER CLICKS NEW CHAT:
#    - Generate new thread_id
#    - Clear display
#    - Ready for first message

# DATABASE PERSISTENCE:
#    - All messages automatically saved by LangGraph checkpointer
#    - Topics saved via bot.update_state()
#    - Everything persists across restarts
#    - No manual database code needed!

# =================================================================================
