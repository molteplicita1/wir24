# Description

This version of the project implements a microservices architecture, designed for enhanced scalability, modularity, and ease of maintenance. 
The functionality is divided into four key microservices, each with a specific role:

- **Gateway:** Acts as the central entry point, receiving and routing all incoming requests to the appropriate microservices.
- **Populate**: Handles the addition of documents to the vector database by calculating their embeddings and storing them.
- **Retrieve**: Processes user queries by calculating their embeddings, using them to retrieve the top-k most relevant documents from the vector database.
- **Generate**: Constructs responses using the context provided by the retrieved documents and the Ollama server running LLMs.

# Docker
## How to Run


1. **Install Ollama**
	
	Download and install Ollama from [here](https://ollama.com/download).

	After the installation, follow this [guide](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server) to expose the Ollama server to the local network
	

2. **Clone the repository** and navigate to the project directory
3. **Configure the .env file**:
   	Create a ```.env``` file in the root of the project with the following parameters to specify the server settings and model configurations:

	```
 	EMBEDDING_MODEL=sample-embed-model
	CHROMA_ADDRESS=chroma
	CHROMA_PORT=8000
	CHROMA_DB=wir
	OLLAMA_ADDRESS=host.docker.internal
	OLLAMA_PORT=11434
 	TEMPERATURE=0.4
 	URL_POPULATE=http://populate:8001/populate
	URL_RETRIEVE=http://retrieve:8002/retrieve
	URL_GENERATE=http://generate:8003/generate
 	```
4. **Download LLM & Embedding Model**:
	```
	ollama pull gemma2:2b
 	```

	```
	ollama pull sample-embed-model
 	```
 

5. Make the ```compose.sh``` script executable:
	  ```
	  chmod 744 compose.sh
	  ```
6. **Execute** the compose.sh script	
	   ```
	   ./compose.sh
	   ```

   The compose.sh script will orchestrate the setup and startup of all microservices, ensuring the system is ready to handle requests.


## Warning  
Before making requests that involve querying Chroma, you must first populate the database by uploading at least one PDF document. Refer to the section [Uploading a document to the database](#uploading-a-document-to-the-database) for instructions.

The requests to **avoid** if no document has been uploaded yet are:  
- [Query for generation](#query-for-generation)  
- [Deleting a document from the database](#deleting-a-document-from-the-database)  
- [Retrieve the list of uploaded documents](#retrieve-the-list-of-uploaded-documents)

The requests that **can** be made regardless are:  
- [Gateway health check](#gateway-health-check)  
- [Health check for other services via the gateway](#health-check-for-other-services-via-the-gateway)  

---

## Requests  
### GET  
#### Gateway health check  
Endpoint: `http://127.0.0.1:8004`  
Example request using curl:  
```shell
curl --location 'http://127.0.0.1:8004'
```

#### Health check for other services via the gateway  
Endpoint: `http://127.0.0.1:8004/services`  
Example request using curl:  
```shell
curl --location 'http://127.0.0.1:8004/services'
```

#### Retrieve the list of uploaded documents  
Endpoint: `http://127.0.0.1:8004/documents`  
Example request using curl:  
```shell
curl --location 'http://127.0.0.1:8004/documents'
```

---

### POST  
#### Uploading a document to the database  
Endpoint: `http://127.0.0.1:8004/document`  
You need to include a PDF document in the request body. 
In Postman, select the "form-data" body type, add a key named `file`, set its type to "File" (default is "Text"), and upload the PDF document.  
Example request using curl:  
```shell
curl --location 'http://127.0.0.1:8004/document' --form 'file=@"path/to/file.pdf"'
```

#### Query for generation  
Endpoint: `http://127.0.0.1:8004/query`  
Include the query in the request body. 
In Postman, select the "raw" body type and provide the query in JSON format. Example:  
```json
{
    "query": "Parlami del pattern singleton"
}
```
Example request using curl:  
```shell
curl --location 'http://127.0.0.1:8004/query' --header 'Content-Type: application/json' --data '{
    "prompt": "Parlami del pattern singleton"
}'
```

---

### DELETE  
#### Deleting a document from the database  
Endpoint: `http://127.0.0.1:8004/document?file_name=***`  
`***` should be replaced with the name of the file to delete.  
Example request using curl:  
```shell
curl --location --request DELETE 'http://127.0.0.1:8004/document?file_name=***'
```
**Tip:** To avoid errors, it is recommended to first retrieve the list of uploaded documents (by making a request to the [/documents](#retrieving-the-list-of-uploaded-documents) endpoint) and copy the name of the file you want to delete.

# Kubernetes
## Coming soon...
