import uuid
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from .storage import save_message, get_history, storage_status
from .engine import ai
from .config import APP_NAME

app = FastAPI(title="LiveVision")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Ensure an upload directory exists for incoming images
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str = Field(min_length=1, max_length=4000)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "app_name": "LiveVision"})

@app.get("/api/status")
async def status():
    return {
        "app": "LiveVision",
        "model": "loaded" if ai.model is not None else "not_loaded",
        "device": "GPU" if __import__("torch").cuda.is_available() else "CPU",
        "storage": storage_status(),
        "cpp_helper": "active" if ai.cpp_helper is not None else "inactive"
    }

@app.get("/api/history/{session_id}")
async def history(session_id: str):
    return {"session_id": session_id, "messages": get_history(session_id)}

@app.post("/api/chat")
async def chat(body: ChatRequest):
    history = get_history(body.session_id)
    save_message(body.session_id, "user", body.message)
    try:
        # Use our C++ helper or standard word count metrics if needed
        word_count = ai.count_text_words(body.message)
        print(f"[LiveVision] Processing prompt with {word_count} words.")
        
        answer = ai.reply(history, body.message)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LiveVision AI model could not start: {exc}")
    save_message(body.session_id, "assistant", answer)
    return {"session_id": body.session_id, "reply": answer}

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Handles image uploads for LiveVision's image-to-script workflow."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")
    
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "success": True, 
        "filename": file.filename, 
        "path": str(file_path),
        "message": "Image successfully uploaded and ready for script generation."
    }

@app.health
@app.get("/health")
async def health():
    return {"ok": True}
  
