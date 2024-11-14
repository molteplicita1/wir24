from functions import delete_doc_in_collection
import chromadb, os

CHROMA_ADDRESS = os.getenv("CHROMA_ADDRESS")
CHROMA_PORT = os.getenv("CHROMA_PORT")
CHROMA_DB = os.getenv("CHROMA_DB")
DATA_PATH = os.getenv("DATA_PATH")
OLLAMA_ADDRESS = os.getenv("OLLAMA_ADDRESS")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
EMBEDDING = os.getenv("EMBEDDING_MODEL")

chroma_client = chromadb.HttpClient(host=CHROMA_ADDRESS, port=CHROMA_PORT)

collection = chroma_client.get_collection(name=f"{CHROMA_DB}")

metadatas = collection.get()['metadatas']
files = set(metadata['source'] for metadata in metadatas)
files = list(files)

print("Choose document to delete:") 
for i, file in enumerate(files):
    print(f"{i}. {file}")

doc_id = int(input("Insert the number of document to delete: "))

if doc_id < 0 or doc_id >= len(files):
    print("Invalid document id.")
    exit()

delete_doc_in_collection(CHROMA_ADDRESS, CHROMA_PORT, CHROMA_DB, files[doc_id])

