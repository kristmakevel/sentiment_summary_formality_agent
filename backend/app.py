from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from fastapi.responses import FileResponse
import os
from backend.agent import run_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    text: str
    instruction: str


@app.post("/process")
async def process_endpoint(request: ProcessRequest):
    result = await asyncio.to_thread(run_agent, request.text, request.instruction)

    return {"result": result}


@app.get("/")
async def serve_index():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(current_dir, "templates", "index.html")

    return FileResponse(file_path)