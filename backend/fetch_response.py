import os 
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI



def get_chatgpt_response(prompt):
    api_key = os.getenv('OPENAI_API_KEY')
    messages = [
        ChatMessage(
            role="system", content="You are his holiness the 14th Dalai Lama."
        ),
        ChatMessage(role="user", content=prompt),
    ]
    response = OpenAI(api_key=api_key).chat(messages)
    answer = response.message.content 
    return answer 


