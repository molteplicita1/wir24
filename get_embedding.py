import json
from langchain_community.embeddings.ollama import OllamaEmbeddings

with open('config.json') as config_file:
    config = json.load(config_file)

def get_embedding_function():
    embeddings = OllamaEmbeddings(model=config['embedding_model'], base_url=config['url'])
    return embeddings
