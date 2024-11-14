import os, time, argparse, chromadb
from functions import populate, pipeline
from tokenizer import TokenizerFactory

CHROMA_ADDRESS = os.getenv("CHROMA_ADDRESS")
CHROMA_PORT = os.getenv("CHROMA_PORT")
CHROMA_DB = os.getenv("CHROMA_DB")
DATA_PATH = os.getenv("DATA_PATH")
OLLAMA_ADDRESS = os.getenv("OLLAMA_ADDRESS")
OLLAMA_PORT = os.getenv("OLLAMA_PORT")
EMBEDDING = os.getenv("EMBEDDING_MODEL")
MODEL = os.getenv("MODEL")

parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="Reset the database.")
args = parser.parse_args()
if args.reset:
    print("✨ Clearing Database")
    chromadb.HttpClient(host=CHROMA_ADDRESS, port=CHROMA_PORT).reset()
    print("Database successfully cleaned! ✅")

tokenizer = TokenizerFactory.create_tokenizer()

print("Populating the database...")
populate(CHROMA_ADDRESS, CHROMA_PORT, CHROMA_DB, DATA_PATH, EMBEDDING, OLLAMA_ADDRESS, OLLAMA_PORT, tokenizer)

with open('../queries.txt') as quer:
    queries = [q.strip("\n") for q in quer]

temperatures = [0.1, 0.4, 0.6]

time_start = time.time()

for temp in temperatures:

    file = f"output_{temp}.md"
    with open(file, 'w') as out:
        out.write("")
        out.close()

    print("Output file cleared")


    with open(file, 'a') as f:
        for query in queries:
            print(f"\nQuery: {query}")
            f.write(f"# QUERY: {query}\n")
            f.flush()
            
            print(f"Starting with {EMBEDDING}, {MODEL} and {type(tokenizer).__name__}")
            f.write(f"## EMBEDDING: {EMBEDDING}\n")
            f.write(f"## MODEL: {MODEL}\n")
            f.write(f"## TOKENIZER: {type(tokenizer).__name__}\n")
            f.flush()

            pipeline(CHROMA_ADDRESS, CHROMA_PORT, CHROMA_DB, EMBEDDING, query, OLLAMA_ADDRESS, OLLAMA_PORT, MODEL, temp)
            print(f"Done with {EMBEDDING}, {MODEL} and {type(tokenizer).__name__}\n")


time_end = time.time()

print(f"Total time: {time_end - time_start} seconds")