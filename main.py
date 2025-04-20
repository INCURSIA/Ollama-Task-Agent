from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from chatbot import chat

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# For POST requests
class Message(BaseModel):
    query: str

@app.post("/chat")
def chat_api(message: Message):
    response = chat(message.query)
    return {"response": response}

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())
