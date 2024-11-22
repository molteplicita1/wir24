from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from utils import add_document, delete_document, list_documents


router = APIRouter(prefix="/populate", tags=["populate"])


@router.get("")
def health_check():
    return {"message": "Populate service is up and running!"}


@router.get("/documents")
async def get_list_documents():
    try:
        documents = list_documents()
        if len(documents) < 0:
            return JSONResponse(
                status_code=400,
                content={"message": "Error in retrieving list of documents"}
            )
        return JSONResponse(
            status_code=200,
            content={"documents": documents},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        result = add_document(document=file)
        if not result:
            return JSONResponse(
                status_code=400,
                content={"message": "Error uploading file"}
            )
        return JSONResponse(
            status_code=200,
            content={"message": "File uploaded with success"},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/delete")
async def delete_file(file_name: str):
    try:
        result = delete_document(document_name=file_name)
        if not result:
            return JSONResponse(
                status_code=400,
                content={"message": "Error in document deletion"}
            )
        return JSONResponse(
            status_code=200,
            content={"message": "Document deleted successfully!"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
