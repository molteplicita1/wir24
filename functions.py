import os
import fitz
import ollama
import re
import json
import requests
import chromadb


def extract_text_from_pdf_files(path):
    text_contents = {}

    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(".pdf"):
                file_path = os.path.join(root, filename)

                with fitz.open(file_path) as pdf_document:
                    content = ""

                    for page_num in range(pdf_document.page_count):
                        page = pdf_document.load_page(page_num)
                        content += page.get_text()

                text_contents[filename] = content

    return text_contents


def chunk_splitter(text, chunk_size=256, overlap=32):
    words = re.findall(r'\S+', text)

    chunks = []
    current_chunk = []
    word_count = 0
    overlap_buffer = []

    for word in words:
        current_chunk.append(word)
        word_count += 1

        if word_count >= chunk_size:
            chunks.append(' '.join(current_chunk))

            overlap_buffer = current_chunk[-overlap:]

            current_chunk = overlap_buffer[:]
            word_count = len(current_chunk)


    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def get_embedding(embedding_model, chunks):
  embeds = ollama.embed(model=embedding_model, input=chunks)
  return embeds.get('embeddings', [])


def populate(chroma_address, chroma_port, chroma_collection, data_path, embedding_model):
    chroma_client = chromadb.HttpClient(host=chroma_address, port=chroma_port)


    print("✨ Clearing Database")
    chroma_client.delete_collection(f"{chroma_collection}_{embedding_model}")
    print(f"Collection ({chroma_collection}_{embedding_model}) deleted successfully! ✅")

    
    collection = chroma_client.get_or_create_collection(name=f"{chroma_collection}_{embedding_model}", metadata={"hnsw:space": "cosine"})

    metadatas = collection.get()['metadatas']
    files = set(metadata['source'] for metadata in metadatas)

    text_data = extract_text_from_pdf_files(data_path)
    
    for file_name, text in text_data.items():

        if file_name in files:
            print(f"Skipping {file_name}, already in the collection.")
            continue

        chunks = chunk_splitter(text)
        embeds = get_embedding(embedding_model, chunks)
        chunk_number = list(range(len(chunks)))
        ids = [file_name + str(index) for index in chunk_number]
        metadatas = [{"source": file_name, "chunk": index} for index in chunk_number]
        print(f"Adding {file_name} (chunks={chunk_number[-1]}) to the collection ({chroma_collection}_{embedding_model})...")
        collection.add(ids=ids, documents=chunks, embeddings=embeds, metadatas=metadatas)

    print(f"Collection ({chroma_collection}_{embedding_model}) populated successfully! ✅")


def retrieve(chroma_address, chroma_port, chroma_collection, embedding_model, query):

    chroma_client = chromadb.HttpClient(host=chroma_address, port=chroma_port)
    collection = chroma_client.get_or_create_collection(name=f"{chroma_collection}_{embedding_model}")

    query_embed = ollama.embed(model=embedding_model, input=query)['embeddings']

    results = collection.query(query_embeddings=query_embed, n_results=5)

    docs = '\n\n'.join(results['documents'][0])

    qualcosa = [f"{metadata['source']}: {metadata['chunk']}" for metadata in results['metadatas'][0]]
    sources = f"{{{', '.join(qualcosa)}}}"


    return docs, sources


def do_query(query, docs, sources, ollama_address, ollama_port, model):
    # Apriamo il file in modalità append in modo da non sovrascrivere ma aggiungere alla fine
    with open("output.md", "a", encoding="utf-8") as f:

        f.write("\n")

        prompt = f"{query} - Rispondi alla domanda in italiano basandoti esclusivamente sui seguenti documenti relativi ad appunti universitari di ingegneria informatica. \nLa risposta deve essere completa, accurata e fornire dettagli rilevanti in relazione al contesto disponibile: \n\n{docs}"

        f.write(f"### Question: \n{prompt}\n")
        f.write("\n\n\n----------------------------------------\n\n\n")
        f.write("Generazione in corso...\n")
        f.write("\n\n\n----------------------------------------\n\n\n")

        url = f"http://{ollama_address}:{ollama_port}/api/generate"

        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload).json().get('response', 'Che dici')

        formatted_response = f"### Response: \n{response}\n\nSources: {sources}\n"
        f.write(formatted_response)
        f.write("\n\n\n----------------------------------------\n\n\n")


def pipeline(chroma_address, chroma_port, chroma_collection, embedding_model, query, ollama_address, ollama_port, model):
    docs, sources = retrieve(chroma_address, chroma_port, chroma_collection, embedding_model, query)
    do_query(query, docs, sources, ollama_address, ollama_port, model)
  
