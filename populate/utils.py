import chromadb
import fitz
import json
import os
import re
import requests


chroma_address = os.getenv("CHROMA_ADDRESS", "localhost")
chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
chroma_collection_name = os.getenv("CHROMA_DB", "wir")
embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
ollama_address = os.getenv("OLLAMA_ADDRESS", "localhost")
ollama_port = os.getenv("OLLAMA_PORT", "11434")


def _tokenize(text):
    text = re.sub(r'Lezione \d+', '', text)
    text = re.sub(r'', '', text)
    text = re.sub(r'\d{2}/\d{2}/\d{4}', '', text)
    return re.findall(r'\S+', text)


def _extract_text_from_pdf(pdf_file) -> str:
    with fitz.open(pdf_file) as pdf_document:
        content = ""

        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            content += page.get_text()
    return content


def _chunk_splitter(tokens, chunk_size=256, overlap=32):
    chunks = []
    current_chunk = []
    token_count = 0
    overlap_buffer = []

    for token in tokens:
        current_chunk.append(token)
        token_count += 1

        if token_count >= chunk_size:
            chunks.append(' '.join(current_chunk))
            overlap_buffer = current_chunk[-overlap:]
            current_chunk = overlap_buffer[:]
            token_count = len(current_chunk)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def _get_embedding(chunks):
    url = f"http://{ollama_address}:{ollama_port}/api/embed"

    payload = json.dumps({
        "model": embedding_model,
        "input": chunks,
        "options": {
            "num_thread": 8
        }
    })

    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload).json()

    return response.get("embeddings")


def delete_document(document_name):
    chroma_client = chromadb.HttpClient(host=chroma_address, port=chroma_port)

    collection = chroma_client.get_collection(name=f"{chroma_collection_name}")

    collection.delete(where={"source": document_name})

    path = "populate/data"
    for file in os.listdir(path):
        if file == document_name:
            os.remove(os.path.join(path, file))
            return True

    return False


def add_document(document):
    print(f"Adding document {document.filename}...")
    if not document.filename.endswith(".pdf"):
        raise ValueError("Uploaded file is not PDF.")

    add_to_dir = _add_to_dir(document)
    if not add_to_dir[0]:
        return False
    _add_to_db(add_to_dir[1])
    return True


def _add_to_dir(document):
    path = "populate/data"
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, document.filename)

    if os.path.exists(file_path):
        return False

    with open(file_path, "wb") as f:
        f.write(document.file.read())

    document.file.seek(0)
    return True, file_path


def _add_to_db(file_path):
    chroma_client = chromadb.HttpClient(host=chroma_address, port=chroma_port)
    collection = chroma_client.get_or_create_collection(name=f"{chroma_collection_name}")
    text = _extract_text_from_pdf(file_path)
    tokens = _tokenize(text)
    chunks = _chunk_splitter(tokens)
    embeds = _get_embedding(chunks)
    chunk_number = list(range(len(chunks)))
    file_name = file_path.split("/")[-1]
    ids = [file_name + str(index) for index in chunk_number]
    metadatas = [{"source": file_name, "chunk": index} for index in chunk_number]
    print(f"Adding {file_name} (chunks={chunk_number[-1]}) to the collection ({chroma_collection_name})...")
    collection.add(ids=ids, documents=chunks, embeddings=embeds, metadatas=metadatas)
    print("Added to collection")


def list_documents():
    chroma_client = chromadb.HttpClient(host=chroma_address, port=chroma_port)
    collection = chroma_client.get_collection(name=f"{chroma_collection_name}")
    metadatas = collection.get()['metadatas']
    files = set(metadata['source'] for metadata in metadatas)
    return list(files)
