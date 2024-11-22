from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils import generate


class Prompt(BaseModel):
    prompt: str
    documents: str
    sources: str
    distance: List[float]
    model: str | None = "gemma2:2b"
    temperature: float | None = 0.4


router = APIRouter(prefix="/generate", tags=["generate"])


@router.get("")
def health_check():
    return {"message": "Generate service is up and running!"}


@router.post("/response")
def generate_response(prompt: Prompt):
    try:
        response = generate(prompt.prompt, prompt.documents, prompt.sources, prompt.distance, prompt.model,
                            prompt.temperature)
        # response = {"response": "This is a test response"}
        return JSONResponse(
            status_code=200,
            content=response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
