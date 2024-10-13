import json
from functions import pipeline

with open('config.json') as f:
    config = json.load(f)

chroma_address = config['chroma_address']
chroma_port = config['chroma_port']
chroma_collection = config['chroma_collection']
data_path = config["data_path"]

ollama_address = config["ollama_address"]
ollama_port = config["ollama_port"]

with open('embeddings.txt') as em:
    embeddings = em.read()

with open('models.txt') as m:
    models = m.read()


query = "Parlami del clustering"


for embedding_model in embeddings:
    for model in models:
        pipeline(chroma_address, chroma_port, chroma_collection, data_path, embedding_model, query, ollama_address, ollama_port, model)

