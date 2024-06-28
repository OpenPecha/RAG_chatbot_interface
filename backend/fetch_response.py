from openai import OpenAI

client = OpenAI()

def get_chatgpt_response(prompt):
    stream = client.chat.completions.create(
         model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are his holiness the 14th Dalai Lama."},
            {"role": "user", "content": prompt}],
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content



