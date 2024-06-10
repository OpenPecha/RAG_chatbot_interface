from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserInput(BaseModel):
    user_input: str


@app.post("/")
async def respond_to_user_input(user_input: UserInput):
    # Placeholder response generation logic
    response = f"Echo from backend: {user_input.user_input}"
    return {"response": response}
