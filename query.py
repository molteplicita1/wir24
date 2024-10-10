import argparse
import requests
import json

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding import get_embedding_function

with open('config.json') as config_file:
    config = json.load(config_file)

CHROMA_PATH = json['chroma_path']


PROMPT_TEMPLATE = """
Rispondi alla domanda in italiano basandoti esclusivamente sui seguenti documenti relativi ad appunti universitari di ingegneria informatica.
La risposta deve essere completa, accurata e fornire dettagli rilevanti in relazione al contesto disponibile.
Gli appunti aggiunti riguardano i corsi:
- Programmazione di Sistemi in Rete
- Laboratorio di elettronica per l'automazione

{context}

---

Rispondi alla seguente domanda (in italiano) seguendo questa struttura: 
1. Introduzione
2. Dettagli principali

Domanda: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)
    

    url = config['url'] + "/api/generate"

    payload = json.dumps({
    "model": config['model'],
    "prompt": prompt,
    "stream": False
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    response_json = response.json()

    response_text = response_json.get('response', 'Che dici')


    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
