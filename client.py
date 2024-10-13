import json
from functions import populate, pipeline

with open('config.json') as f:
    config = json.load(f)

chroma_address = config['chroma_address']
chroma_port = config['chroma_port']
chroma_collection = config['chroma_collection']
data_path = config["data_path"]

ollama_address = config["ollama_address"]
ollama_port = config["ollama_port"]

with open('embeddings.txt') as em:
    embeddings = [e.strip() for e in em]


with open('models.txt') as mod:
    models = [m.strip() for m in mod]


query = "Parlami del clustering"


for embedding_model in embeddings:
    print(embedding_model)
    populate(chroma_address, chroma_port, chroma_collection, data_path, embedding_model)
    for model in models:
        for i in range(0, 2):
            pipeline(chroma_address, chroma_port, chroma_collection, embedding_model, query, ollama_address, ollama_port, model)

