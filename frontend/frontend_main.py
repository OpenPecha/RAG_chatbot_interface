import requests
import streamlit as st

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
    response = requests.post("http://127.0.0.1:8000", json={"user_input": user_input}, stream=True)
    return response.iter_lines()

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_response_from_backend(prompt)
    assistant_message = st.chat_message("assistant")
    response_text = ""

    for line in response:
        if line:
            chunk = line.decode('utf-8')
            response_text += chunk 
            assistant_message.markdown(chunk)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
