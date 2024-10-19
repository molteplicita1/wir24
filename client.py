from tokenizer import TokenizerFactory
from functions import *
import os, argparse


with open('output.md', 'w') as f:
    f.write("")
    f.close()

CHROMA_ADDRESS = os.getenv("CHROMA_ADDRESS")
CHROMA_PORT = os.getenv("CHROMA_PORT")
CHROMA_DB = os.getenv("CHROMA_DB")
DATA_PATH = os.getenv("DATA_PATH")
TOKENIZER = os.getenv("TOKENIZER")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MODEL = os.getenv("MODEL")
OLLAMA_ADDRESS = os.getenv("OLLAMA_ADDRESS")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")

parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="Reset the database.")
args = parser.parse_args()
if args.reset:
    print("✨ Clearing Database")
    chromadb.HttpClient(host=CHROMA_ADDRESS, port=CHROMA_PORT).reset()
    print("Database successfully cleaned! ✅")

collection_name = f"{CHROMA_DB}"

tokenizer = TokenizerFactory.create_tokenizer()

populate(CHROMA_ADDRESS, CHROMA_PORT, collection_name, DATA_PATH, EMBEDDING_MODEL, OLLAMA_ADDRESS, OLLAMA_PORT, tokenizer)

query = "Parlami del clustering che abbiamo studiato in Data Analytics"

docs, source, distances = retrieve(CHROMA_ADDRESS, CHROMA_PORT, collection_name, EMBEDDING_MODEL, query, OLLAMA_ADDRESS, OLLAMA_PORT)

do_query(query, docs, source, distances, OLLAMA_ADDRESS, OLLAMA_PORT, MODEL)


print("\nDone!")