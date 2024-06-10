TEXT_GENERATION_ARGS = {
        "max_new_tokens": 3000,
        "return_full_text": False,
        "temperature": 0,
        "do_sample": False,
    }


EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"


db_path="./chroma_db"
collection_name="rag_demo"
persist_dir="./chroma_db/index"