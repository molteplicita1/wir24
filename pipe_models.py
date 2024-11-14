import os, time, argparse, chromadb
from functions import populate, pipeline
from tokenizer import TokenizerFactory

CHROMA_ADDRESS = os.getenv("CHROMA_ADDRESS")
CHROMA_PORT = os.getenv("CHROMA_PORT")
CHROMA_DB = os.getenv("CHROMA_DB")
DATA_PATH = os.getenv("DATA_PATH")
TOKENIZER = os.getenv("TOKENIZER")
OLLAMA_ADDRESS = os.getenv("OLLAMA_ADDRESS")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")

parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="Reset the database.")
args = parser.parse_args()
if args.reset:
    print("✨ Clearing Database")
    chromadb.HttpClient(host=CHROMA_ADDRESS, port=CHROMA_PORT).reset()
    print("Database successfully cleaned! ✅")

simple_tokenizer = TokenizerFactory.create_tokenizer()
gemma_tokenizer = TokenizerFactory.create_tokenizer(model=TOKENIZER)

tokenizers = [simple_tokenizer, gemma_tokenizer]


with open('../embeddings.txt') as em:
    embeddings = [e.strip() for e in em]


with open('../models.txt') as mod:
    models = [m.strip() for m in mod]


with open('output.md', 'w') as out:
    out.write("")
    out.close()


print("Output file cleared")

with open('../queries.txt') as quer:
    queries = [q.strip("\n") for q in quer]


for embedding_model in embeddings:
    for tokenizer in tokenizers:
        for model in models:
            collection_name = f"{CHROMA_DB}_{embedding_model}_{type(tokenizer).__name__}"
            populate(CHROMA_ADDRESS, CHROMA_PORT, collection_name, DATA_PATH, embedding_model, OLLAMA_ADDRESS, OLLAMA_PORT, tokenizer)

 
print("\n")

time_start = time.time()

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
                    collection_name = f"{CHROMA_DB}_{embedding_model}_{type(tokenizer).__name__}"
                    pipeline(CHROMA_ADDRESS, CHROMA_PORT, collection_name, embedding_model, query, OLLAMA_ADDRESS, OLLAMA_PORT, model)
                    print(f"Done with {embedding_model}, {model} and {type(tokenizer).__name__}\n")
        f.write("\n\n\n----------------------------------------\n\n\n")

time_end = time.time()

print(f"Total time: {time_end - time_start} seconds")