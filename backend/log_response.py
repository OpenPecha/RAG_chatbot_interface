from pathlib import Path

LOG_FILE_PATH = Path("rag_chatbot_log.txt")

def log_rag_chatbot_response(question:str, response:str):
    if not LOG_FILE_PATH.exists():
        LOG_FILE_PATH.touch()


    question = question.replace("\n","")
    response = response.replace("\n","")

    """ response contains both answer and references"""
    """ we will store them separelty in log file"""
    response_parts = response.split("__References__")
    answer, references = response_parts[0], response_parts[1]
    if answer.startswith("Answer:"):
        answer = answer.removeprefix("Answer:")
        
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as file:
        file.write(f"Question: {question},Answer:{answer},References:{references}\n")
