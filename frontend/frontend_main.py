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
def get_response_from_backend(user_input: str) -> str:
    response = requests.post("http://127.0.0.1:8000", json={"user_input": user_input})
    if response.status_code == 200:
        return response.json().get("response", "Error: No response from backend")
    else:
        return "Error: Failed to connect to backend"

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_response_from_backend(prompt)
    answer, answer_references = response 
    # Display assistant response in chat message container
    
    output = ""
    with st.chat_message("assistant"):
        st.markdown(answer)
        output += answer + "\n\n"
        if answer_references and answer != "I dont have enough data to provide an answer.":
            st.markdown("__References__")
            output += "__References__\n\n"
            for idx, reference in enumerate(answer_references, start=1):
                citation = f"_{idx}. Book Title: {reference['book_title']}, Page No: {reference['page_no']}, Chapter: {reference['chapter']}_"
                st.markdown(citation)
                output += citation + "\n\n"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": output})