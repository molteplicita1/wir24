import os
from typing import List

import requests
import httpx

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel


url_populate = os.getenv("URL_POPULATE", "http://127.0.0.1:8001/populate")
url_retrieve = os.getenv("URL_RETRIEVE", "http://127.0.0.1:8002/retrieve")
url_generate = os.getenv("URL_GENERATE", "http://127.0.0.1:8003/generate")


class Query(BaseModel):
    query: str
    n_results: int | None = 3


class Prompt(BaseModel):
    prompt: str
    documents: str
    sources: str
    distance: List[float]
    model: str | None = "gemma2:2b"
    temperature: float | None = 0.4


class MakeRequest(BaseModel):
    prompt: str
    model: str | None = "gemma2:2b"
    temperature: float | None = 0.4


app = FastAPI()

MICROSERVICES = {
    "populate": url_populate,
    "retrieve": url_retrieve,
    "generate": url_generate
}


@app.get("/")
async def health_check():
    return {"message": "Gateway is up and running!"}


@app.get("/services")
async def health_services():
    results = {}
    for service, url in MICROSERVICES.items():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}")
                results[service] = response.json()
        except Exception as e:
            results[service] = {"error": str(e)}
    return results


@app.get("/documents")
async def get_list_documents():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url_populate}/documents")
            if response.status_code != 200:
                return JSONResponse(
                    status_code=400,
                    content={"message": "Error in retrieving list of documents"}
                )
            return JSONResponse(
                status_code=200,
                content=response.json(),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/document")
def upload_document(file: UploadFile = File(...)):
    try:
        file_content = file.file.read()

        response = requests.post(
            f"{url_populate}/upload",
            files={"file": (file.filename, file_content, file.content_type)}
        )

        if response.status_code != 200:
            return JSONResponse(
                status_code=400,
                content={"message": "Error in retrieving list of documents"}
            )
        return JSONResponse(
            status_code=200,
            content=response.json()
        )
    except Exception as e:
        print(f"Exception: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.delete("/document")
async def delete_document(file_name: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{url_populate}/delete?file_name={file_name}")
            if response.status_code != 200:
                return JSONResponse(
                    status_code=400,
                    content={"message": "Error in retrieving list of documents"}
                )
            return JSONResponse(
                status_code=200,
                content=response.json(),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/query")
async def do_query(make_request: MakeRequest):
    try:
        async with httpx.AsyncClient() as client:
            query = Query(query=make_request.prompt)
            print("Retrieving documents...")
            retrieve = await client.post(f"{url_retrieve}/top", json=query.model_dump())
            if retrieve.status_code != 200:
                return JSONResponse(
                    status_code=400,
                    content={"message": "Error in retrieving documents"}
                )

            documents = retrieve.json().get("documents")
            sources = retrieve.json().get("sources")
            print(f"Sources: {sources}")
            distance = retrieve.json().get("distance")

            prompt = Prompt(
                prompt=make_request.prompt,
                documents=documents,
                sources=sources,
                distance=distance,
                model=make_request.model,
                temperature=make_request.temperature
            )

            print("Generating response...")
            generate = requests.post(f"{url_generate}/response", json=prompt.model_dump())
            if generate.status_code != 200:
                return JSONResponse(
                    status_code=400,
                    content={"message": "Error in generating"}
                )

            print("Response generated")
            return JSONResponse(
                status_code=200,
                content=generate.json()
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
