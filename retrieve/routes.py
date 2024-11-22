from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils import retrieve


class Query(BaseModel):
    query: str
    n_results: int | None = 3


router = APIRouter(prefix="/retrieve", tags=["retrieve"])


@router.get("")
def health_check():
    return {"message": "Retrieve service is up and running!"}


@router.post("/top")
def retrieve_top(query: Query):
    try:
        docs, sources, distance = retrieve(query.query, query.n_results)
        return JSONResponse(
            status_code=200,
            content={"documents": docs, "sources": sources, "distance": distance}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
