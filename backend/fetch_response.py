from openai import OpenAI
from llama_index.core import PromptTemplate
from typing import List, Dict 


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

def transform_query(query:str, older_conversation: List[Dict]):
    """
    transforming the query to make it more clear and concise
    Example:
    query = 'Who was songtsen gampo?'
    llm response = 'Songtsen gampo was a Tibetan Kings,.....'
    
    query 2 = 'who were his wife?' (After query transformation = 'Who were the wives of Songtsen Gampo?') 
    """
    
    template = f"""
        
       
        Question: {query}
        Transform the query such that the question is clear and concise.
    """
    
    qa_template = PromptTemplate(template)
    prompt = qa_template.format(question=query)
    
    """ include the older conversation to make the model understand the context of the query transformation"""
    messages = older_conversation
    messages.extend([{"role": "user", "content": prompt}])

    response = client.chat.completions.create(
         model="gpt-4-turbo",
        messages=messages,
        temperature=0.3,
        
    )
    transformed_query = response.choices[0].message.content
    return transformed_query
    



