# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn, os
from typing import Optional

# Import your multimodal_chat function from your module or paste it here
override_base = os.getcwd()
os.makedirs("uploads", exist_ok=True)

from chatbot import multimodal_chat  # Replace with actual module name

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/analyze")
async def analyze(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    image_path = None
    if file and file.filename != "":
        image_path = os.path.join("uploads", file.filename)
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

    response = multimodal_chat(input_text=text, image_path=image_path)
    return JSONResponse(content={"response": response})
@app.post("/delete-memory")
async def delete_memory():
    memory_file = "memory.json"
    if os.path.exists(memory_file):
        os.remove(memory_file)
        return {"message": "Memory file deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="Memory file not found.")
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)