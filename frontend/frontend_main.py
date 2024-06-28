import requests
import streamlit as st

from log_response import log_rag_chatbot_response

st.set_page_config(page_title="RAG chatbot")
with st.sidebar:
    st.title('RAG Chatbot')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get response from the backend
def get_response_from_backend(user_input: str):
    for chunk in requests.post("http://127.0.0.1:8000", json={"user_input": user_input}, stream=True):
        yield chunk.decode("utf-8") 

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    full_response = ""

    with st.chat_message("assistant"):
        response = get_response_from_backend(prompt)
        full_response = st.write_stream(response)

    log_rag_chatbot_response(prompt, full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
