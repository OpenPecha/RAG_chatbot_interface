import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


EMBEDDING_MODEL = "openpecha/Finetuned_Alibaba_Large"

db_path="./fine_tunned_context_db"
collection_name="rag_demo"
persist_dir="./fine_tunned_context_db/index"

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



GIBBERISH_QUERY_RESPONSE = "It looks like your message is a bit unclear. Could you please rephrase your query so I can better understand and assist you?"

INAPPROPRIATE_QUERY_RESPONSE = "Your query contains language that is inappropriate or harmful. Please rephrase your question in a respectful manner. If you have any legitimate questions or need assistance on a relevant topic, feel free to ask."


NON_ENGLISH_QUERY_RESPONSE = "Please note that we only support queries in English. Kindly rephrase your question in English so we can assist you better."
    