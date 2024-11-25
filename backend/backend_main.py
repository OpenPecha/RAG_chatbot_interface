import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from config import load_vector_db
from fetch_response import get_answer_for_query, transform_query
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

    
    for output in get_answer_for_query(query=query, context=context):
        yield output 
