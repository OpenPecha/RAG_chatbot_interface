from openai import OpenAI

client = OpenAI()

def get_chatgpt_response(prompt):
    stream = client.chat.completions.create(
         model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            # print(chunk.choices[0].delta.content, end="")
            yield chunk.choices[0].delta.content



