import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from llama_index.core import PromptTemplate
from config import load_vector_db
from fetch_response import get_chatgpt_response
from log_response import log_rag_chatbot_response
from fastapi.responses import StreamingResponse

class UserInput(BaseModel):
    user_input: str

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
    return StreamingResponse(generate_answer(user_input.user_input), media_type='text/plain')



def get_threshold(similarity_scores) -> int:
    threshold = np.max(similarity_scores) - np.std(similarity_scores)
    return threshold

def generate_answer(question, num_of_context=10):
    question = question.replace('"', "'")
    retriever = VECTOR_INDEX.as_retriever(similarity_top_k=num_of_context)
    retrieved_nodes = retriever.retrieve(question)

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

    template = f"""
        You are his holiness the 14th Dalai Lama.
        
        Strictly follow these guidelines when answering the questions:
        
        - Answer the question based on the given context (some of which might be irrelevant).
        - Provide a short, informative, and pleasant response.
        - Use plain and respectful language.
        - If there is not enough information in the context to answer the question, respond with "I don't have enough data to provide an answer."

        Question: {question}
        {context}
        
        Your task is divided into two parts:
        
        1. **Get the Answer:**
        - Provide a concise and precise answer to the user's question based on the given context.
        
        2. **Find the Source Snippets:**
        - Extract and provide all relevant snippets from the contexts that directly support your answer.
        - Ensure each snippet retains the exact wording and spelling from the context.
        - Cite the source of each snippet (e.g., book title, page number, chapter) in italic.
        - Snippet from same source must be shown together.
        - Separate each snippet and its source with a new line.

        Structure your response as follows:
        
        your_answer
        
        __References__
        1._source_:snippet_1   

        2._source_:snippet_2   
        
        ...
    """
    
    qa_template = PromptTemplate(template)
    prompt = qa_template.format(context=context, question=question)
    
    for output in get_chatgpt_response(prompt):
        yield output 
