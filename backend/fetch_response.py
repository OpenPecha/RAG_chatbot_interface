

import requests

def get_chatgpt_response(api_key, prompt):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4-turbo", 
        "messages": [
            {"role": "system", "content": "You are his holiness the 14th Dalai Lama."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    return response_json
