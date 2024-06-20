from pathlib import Path

LOG_FILE_PATH = Path("rag_chatbot_log.txt")

def log_rag_chatbot_response(question:str, answer:str):
    if not LOG_FILE_PATH.exists():
        LOG_FILE_PATH.touch()


    question = question.replace("\n","")
    answer = answer.replace("\n","")
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as file:
        file.write(f"Question: {question},Answer:{answer}\n")
