# RAG_chatbot_interface

An AI-powered chatbot using Retrieval-Augmented Generation (RAG) to provide accurate answers from the Dalai Lama's books.

Data preparation, including vectorization, has been done using [rag prep tool](https://github.com/OpenPecha/rag_prep_tool).

The fine-tuned embedding model is available and has been uploaded to [OpenPecha's Hugging Face](https://huggingface.co/openpecha/Finetuned_Alibaba_Large).


## Installation
```bash
# Clone the repository
git clone https://github.com/OpenPecha/RAG_chatbot_interface.git

# Navigate to the project directory
cd RAG_chatbot_interface

# Install the required dependencies
pip install -r requirements.txt
```


## Starting the Chatbot
The frontend is built using Streamlit. Start it by running:

```bash
streamlit run frontend/frontend_main.py
```

The backend is built using FastAPI. Start it by running:
```bash
fastapi dev backend/backend_main.py
```

## Important Notes
Make sure to set your OpenAI API key in the environment variables before starting the chatbot.



