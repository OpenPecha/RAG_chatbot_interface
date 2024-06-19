import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from llama_index.core import PromptTemplate

from config import load_vector_db
from fetch_response import get_chatgpt_response
class UserInput(BaseModel):
    user_input: str


VECTOR_INDEX = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global VECTOR_INDEX
    VECTOR_INDEX = load_vector_db()
    yield

    # Shut down code goes here
    print("Shut Down")


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the API. Please use the POST method to get responses."}


@app.post("/")
async def respond_to_user_input(user_input: UserInput):
    # Placeholder response generation logic
    response = generate_answer(user_input.user_input)
    return {"response": response}

def get_threshold(similarity_scores)->int:
    threshold = np.max(similarity_scores) - np.std(similarity_scores) 
    return threshold

def generate_answer(question, num_of_context=10)->str:
    question = question.replace('"',"'")
    retriever = VECTOR_INDEX.as_retriever(similarity_top_k=num_of_context)
    retrieved_nodes = retriever.retrieve(question)

    context = ""
    
    similarity_scores = [retrieved_node.score for retrieved_node in retrieved_nodes]
    threshold = get_threshold(similarity_scores)

    answer_references = []

    keys_to_copy = ['book_title', 'page_no', 'chapter']

    for retrieved_node in retrieved_nodes:
        if retrieved_node.score < threshold or retrieved_node.score < 0 :
            continue
        context += f"{retrieved_node.get_content()} \n\n"
        
        curr_reference = {key: retrieved_node.metadata[key] for key in keys_to_copy}
        answer_references.append(curr_reference)
    
    template = f"""
    You are his holiness the 14th Dalai Lama.
    
    Strictly follow these guidelines when answering the questions:
    
    - Answer the question based on the given contexts (some of which might be irrelevant).
    - Give me a short but informative and pleasant answer.
    - Speak in plain English.
    - Be careful of the language, ensuring it is respectful and appropriate.
    - If you do not have a proper answer from the context, only respond with "I dont have enough data to provide an answer."
    
    Question: {question}
    Contexts: {context}
    
    
    """

    qa_template = PromptTemplate(template)
    prompt = qa_template.format(context=context, question=question)
    """ Get the response from the chatgpt 4 model """
    answer = get_chatgpt_response(prompt)
    return (answer, answer_references)


