import os, fitz, json, requests, chromadb

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


def chunk_splitter(tokens, chunk_size=256, overlap=32):
    
    chunks = []
    current_chunk = []
    token_count = 0
    overlap_buffer = []

    for token in tokens:
        current_chunk.append(token)
        token_count += 1

        if token_count >= chunk_size:
            # Se il conteggio delle parole è maggiore o uguale alla dimensione del chunk desiderata (chunk_size),
            # si procederà a creare un nuovo chunk.

            # Aggiunge il chunk corrente (current_chunk) alla lista dei chunks.
            # Il chunk è una lista di parole, quindi viene unita in una stringa usando ' '.join(),
            # ovvero unisce le parole della lista in una singola stringa separata da spazi.
            chunks.append(' '.join(current_chunk))

            # Prende le ultime "overlap" parole del chunk corrente (overlap è un numero predefinito),
            # e le assegna a overlap_buffer. Questo viene fatto per creare una sovrapposizione tra i chunk
            # (una sorta di continuità tra le sezioni di testo).
            overlap_buffer = current_chunk[-overlap:]

            # Imposta current_chunk uguale a overlap_buffer. Questo resetta current_chunk,
            # mantenendo solo le parole sovrapposte dall'ultimo chunk.
            # In pratica, si prepara per iniziare un nuovo chunk, iniziando con le parole sovrapposte.
            current_chunk = overlap_buffer[:]

            # Aggiorna il contatore di parole (word_count) con la lunghezza del nuovo current_chunk,
            # che ora contiene solo le parole dell'overlap, in modo da prepararsi a riempirlo nuovamente.
            token_count = len(current_chunk)


    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def get_embedding(embedding_model, chunks, ollama_address, ollama_port):

    # chiamata http
    url = f"http://{ollama_address}:{ollama_port}/api/embed"
    
    payload = json.dumps({
        "model": embedding_model,
        "input": chunks,
        "options": {
            "num_thread": 8
        }
    })

    headers={'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers = headers, data=payload).json()

    return response.get("embeddings")


def populate(chroma_address, chroma_port, chroma_collection_name, data_path, embedding_model, ollama_address, ollama_port, tokenizer):
    chroma_client = chromadb.HttpClient(host=chroma_address, port=chroma_port)
    
    collection = chroma_client.get_or_create_collection(name=f"{chroma_collection_name}", metadata={"hnsw:space": "cosine"})

    metadatas = collection.get()['metadatas']
    files = set(metadata['source'] for metadata in metadatas)

    text_data = extract_text_from_pdf_files(data_path)
    
    for file_name, text in text_data.items():

        if file_name in files:
            print(f"Skipping {file_name}, already in the collection.")
            continue

        tokens = tokenizer.tokenize(text)
        chunks = chunk_splitter(tokens)
        embeds = get_embedding(embedding_model, chunks, ollama_address, ollama_port)
        chunk_number = list(range(len(chunks)))
        ids = [file_name + str(index) for index in chunk_number]
        metadatas = [{"source": file_name, "chunk": index} for index in chunk_number]
        print(f"Adding {file_name} (chunks={chunk_number[-1]}) to the collection ({chroma_collection_name})...")
        collection.add(ids=ids, documents=chunks, embeddings=embeds, metadatas=metadatas)

    print(f"Collection ({chroma_collection_name}) populated successfully! ✅")


def retrieve(chroma_address, chroma_port, chroma_collection_name, embedding_model, query, ollama_address, ollama_port):

    chroma_client = chromadb.HttpClient(host=chroma_address, port=chroma_port)
    collection = chroma_client.get_or_create_collection(name=f"{chroma_collection_name}")

    query_embed = get_embedding(embedding_model, [query], ollama_address, ollama_port)

    results = collection.query(query_embeddings=query_embed, n_results=3)

    docs = '\n\n'.join(results['documents'][0])

    distance = results['distances'][0]

    something = [f"{metadata['source']}: {metadata['chunk']}" for metadata in results["metadatas"][0]]
    sources = f"{{{', '.join(something)}}}"

    return docs, sources, distance


def do_query(query, docs, sources, distance, ollama_address, ollama_port, model, temperature=0.8):
    # Apriamo il file in modalità append in modo da non sovrascrivere ma aggiungere alla fine
    with open(f"output_{temperature}.md", "a", encoding="utf-8") as f:

        f.write("\n")

        prompt = f"{query} - Rispondi alla domanda in italiano basandoti ESCLUSIVAMENTE sui seguenti documenti relativi ad appunti universitari di ingegneria informatica. \nLa risposta deve essere completa, accurata e fornire dettagli rilevanti in relazione al contesto disponibile. \nNota bene: se nel contesto seguente ci sono degli esempi, non considerarli per fornire la spiegazione.\nIl contesto è:\n\n{docs}"

        f.write(f"### Question: \n{prompt}\n")
        f.write("\n\n\n----------------------------------------\n\n\n")
        f.write("Generazione in corso...\n")
        f.write("\n\n\n----------------------------------------\n\n\n")

        url = f"http://{ollama_address}:{ollama_port}/api/generate"

        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options":{
                "num_thread": 8,
                "temperature": temperature
            }
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload).json().get('response', 'Che dici')

        formatted_response = f"### Response: \n{response}\n\nSources: {sources}\n\n Distance: {distance}\n"
        f.write(formatted_response)
        f.write("\n\n\n----------------------------------------\n\n\n")


def pipeline(chroma_address, chroma_port, chroma_collection_name, embedding_model, query, ollama_address, ollama_port, model, temperature=0.8):
    docs, sources, distance = retrieve(chroma_address, chroma_port, chroma_collection_name, embedding_model, query, ollama_address, ollama_port)
    do_query(query, docs, sources, distance, ollama_address, ollama_port, model, temperature)
  
