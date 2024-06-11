from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager


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



#####################################################################################
#                     LOAD VECTOR DATABASE                     #
#####################################################################################

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, load_index_from_storage, Settings
from config import db_path, collection_name, persist_dir, EMBEDDING_MODEL

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import PromptTemplate

def load_vector_db():
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL,trust_remote_code=True)
    Settings.embed_model = embed_model

    """ Initialize the Chroma persistent client """
    chroma_client = chromadb.PersistentClient(path=db_path)

    """ Get or create the specified collection"""
    chroma_collection = chroma_client.get_or_create_collection(collection_name)

    """ Initialize the vector store with the specified collection """
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    """ Initialize the storage context with the vector store and persist directory """
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)

    """ Load the index from storage """
    VECTOR_INDEX = load_index_from_storage(storage_context)
    return VECTOR_INDEX


def generate_answer(question, num_of_context=5)->str:
    question = question.replace('"',"'")
    retriever = VECTOR_INDEX.as_retriever(similarity_top_k=num_of_context)
    retrieved_nodes = retriever.retrieve(question)

    context = ""

    
    for retrieved_node in retrieved_nodes:
        context += f"{retrieved_node.get_content()} \n\n"
    
    template = f"""
    You are a chatbot designed to answer questions using content from the Dalai Lama's books.
    
    Follow these guidelines:
    
    - Answer the question based on the given contexts (some of which might be irrelevant).
    - Be elaborate and precise.
    - Answer directly, without adding any extra words.
    - Be careful of the language, ensuring it is respectful and appropriate.
    - If you do not have a proper answer from the context, respond with "I dont have enough data to provide an answer."
    - Do not give a response longer than 3000 tokens.
    
    Question: {question}
    Contexts: {context}
    
    
    """

    qa_template = PromptTemplate(template)
    prompt = qa_template.format(context_str=context, question_str=question)
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    # output = pipe(messages, **TEXT_GENERATION_ARGS)
    # answer = output[0]['generated_text']
    return context[:200]

