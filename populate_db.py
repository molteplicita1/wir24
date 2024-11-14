import os
from functions import populate
from tokenizer import TokenizerFactory

CHROMA_ADDRESS = os.getenv("CHROMA_ADDRESS")
CHROMA_PORT = os.getenv("CHROMA_PORT")
CHROMA_DB = os.getenv("CHROMA_DB")
DATA_PATH = os.getenv("DATA_PATH")
OLLAMA_ADDRESS = os.getenv("OLLAMA_ADDRESS")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
EMBEDDING = os.getenv("EMBEDDING_MODEL")

tokenizer = TokenizerFactory.create_tokenizer()

populate(CHROMA_ADDRESS, CHROMA_PORT, CHROMA_DB, DATA_PATH, EMBEDDING, OLLAMA_ADDRESS, OLLAMA_PORT, tokenizer)

