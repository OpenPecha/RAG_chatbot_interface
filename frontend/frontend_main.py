import requests
import streamlit as st
from typing import List, Dict 

from log_response import log_rag_chatbot_response


CHATGPT_TOKEN_LIMIT = 1000


def get_token_used():
    token_used = 0
    for message in st.session_state.messages:
        token_used += len(message["content"].split(" "))
    return token_used

st.set_page_config(page_title="RAG chatbot")
with st.sidebar:
    st.title('RAG Chatbot')

with st.container():
    # Token usage progress bar
    
    if "messages" not in st.session_state:
        token_used = 0

    else:
        token_used = get_token_used()

    progress = int((token_used/CHATGPT_TOKEN_LIMIT) * 100)

    st.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
    # Create columns for side-by-side layout
    col1, col2 = st.columns([2, 8])  # Adjust the ratio as needed
    with col1:
        st.write("**Token usage**")

    with col2:
        progress_bar = st.progress(progress)

st.markdown(
    """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 2.875rem;
            background-color: white;
            z-index: 999;
        }
        .fixed-header {
        }
    </style>
        """,
        unsafe_allow_html=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get response from the backend
def get_response_from_backend(user_input: str, older_conversation: List[Dict]):
    for chunk in requests.post("http://127.0.0.1:8000", json={"user_input": user_input, "older_conversation":older_conversation}, stream=True):
        yield chunk.decode("utf-8") 

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    full_response = ""

    with st.chat_message("assistant"):
        response = get_response_from_backend(prompt, st.session_state.messages)
        full_response = st.write_stream(response)

    log_rag_chatbot_response(prompt, full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    token_used = get_token_used()

    progress = int((token_used/CHATGPT_TOKEN_LIMIT) * 100)
    progress_bar.progress(progress)


