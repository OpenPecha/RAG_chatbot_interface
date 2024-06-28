from openai import OpenAI
from llama_index.core import PromptTemplate

client = OpenAI()

def get_answer_for_query(query:str, context:str):

    template = f"""
        
        Strictly follow these guidelines when answering the questions:
        
        - Answer the question based on the given context (some of which might be irrelevant).
        - Provide a short, informative, and pleasant response.
        - Use plain and respectful language.
        - If there is not enough information in the context to answer the question, respond with "I don't have enough data to provide an answer."

        Question: {query}
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
    prompt = qa_template.format(context=context, question=query)
    
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


    