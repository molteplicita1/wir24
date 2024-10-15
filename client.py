import json, time
from functions import populate, pipeline
from tokenizer import TokenizerFactory

time_start = time.time()

t1 = TokenizerFactory.create_tokenizer()
t2 = TokenizerFactory.create_tokenizer(model="google/gemma-2-2b")

tokenizers = [t1, t2]

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


with open('output.md', 'w') as out:
    out.write("")
    out.close()


print("Output file cleared")

with open('queries.txt') as quer:
    queries = [q.strip("\n") for q in quer]

for embedding_model in embeddings:
    for tokenizer in tokenizers:
        populate(chroma_address, chroma_port, chroma_collection, data_path, embedding_model, ollama_address, ollama_port, tokenizer)

print("\n")

with open('output.md', 'a') as f:
    for tokenizer in tokenizers:
        for query in queries:
            print(f"\nQuery: {query}")
            f.write(f"# QUERY: {query}\n")
            f.flush()
            for embedding_model in embeddings:
                for model in models:
                    print(f"Starting with {embedding_model}, {model} and {type(tokenizer).__name__}")
                    f.write(f"## EMBEDDING: {embedding_model}\n")
                    f.write(f"## MODEL: {model}\n")
                    f.write(f"## TOKENIZER: {type(tokenizer).__name__}\n")
                    f.flush()
                    pipeline(chroma_address, chroma_port, chroma_collection, embedding_model, query, ollama_address, ollama_port, model)
                    print(f"Done with {embedding_model}, {model} and {type(tokenizer).__name__}\n")
        f.write("\n\n\n----------------------------------------\n\n\n")

time_end = time.time()

print(f"Total time: {time_end - time_start} seconds")