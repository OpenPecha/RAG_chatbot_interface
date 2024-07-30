import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from config import load_vector_db, GIBBERISH_QUERY_ANSWER, INAPPROPRIATE_QUERY_ANSWER, NON_ENGLISH_QUERY_ANSWER
from fetch_response import classify_query, transform_query, get_answer_for_genuine_query,  get_answer_for_normal_conversation 
from fastapi.responses import StreamingResponse
from typing import List, Dict
class UserInput(BaseModel):
    user_input: str
    older_conversation: List[Dict]

VECTOR_INDEX = None

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global VECTOR_INDEX
    VECTOR_INDEX = load_vector_db()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the API. Please use the POST method to get responses."}

@app.post("/")
async def respond_to_user_input(user_input: UserInput):
    return StreamingResponse(generate_answer(user_input.user_input, user_input.older_conversation), media_type='text/plain')



def get_threshold(similarity_scores) -> int:
    threshold = np.max(similarity_scores) - np.std(similarity_scores)
    return threshold

def generate_answer(query:str, older_conversation, num_of_context=10):
    query = query.replace('"', "'")
    query_category = classify_query(query)
    if query_category == "Gibberish":
        for answer in GIBBERISH_QUERY_ANSWER:
            yield answer
        return 

    if query_category == "Inappropriate":
        for answer in INAPPROPRIATE_QUERY_ANSWER:
            yield answer
        return 
    
    if query_category == "Non-English":
        for answer in NON_ENGLISH_QUERY_ANSWER:
            yield answer
        return
    
    if query_category == "Normal Conversation":
        for answer in  get_answer_for_normal_conversation(query):
            yield answer 
        return 

    """ if the query is a genuine query, we will transform the query to make it more clear and concise"""

    query = transform_query(query, older_conversation)
    
    retriever = VECTOR_INDEX.as_retriever(similarity_top_k=num_of_context)
    retrieved_nodes = retriever.retrieve(query)

    context = ""
    similarity_scores = [retrieved_node.score for retrieved_node in retrieved_nodes]
    threshold = get_threshold(similarity_scores)

    keys_to_copy = ['book_title', 'page_no', 'chapter']
    for retrieved_node in retrieved_nodes:
        if retrieved_node.score < threshold or retrieved_node.score < 0:
            continue
        curr_reference = {key: retrieved_node.metadata[key] for key in keys_to_copy}
        context_metadata = ', '.join([f"{key}: {value}" for key, value in curr_reference.items()])
        context += f"Source: {context_metadata}\nSource context:{retrieved_node.get_content()} \n\n"

    
    for output in get_answer_for_genuine_query(query=query, context=context):
        yield output 
