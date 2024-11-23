## Description

This version of the system implements a microservices architecture, designed for enhanced scalability, modularity, and ease of maintenance. 
The functionality is divided into four key microservices, each with a specific role:

- **Gateway:** Acts as the central entry point, receiving and routing all incoming requests to the appropriate microservices.
- **Populate**: Handles the addition of documents to the vector database by calculating their embeddings and storing them.
- **Retrieve**: Processes user queries by calculating their embeddings, using them to retrieve the top-k most relevant documents from the vector database.
- **Generate**: Constructs responses using the context provided by the retrieved documents and the Ollama server running LLMs.

## How to Run


1. **Install Ollama**
	
	Download and install Ollama from [here](https://ollama.com/download).

	After the installation, follow this [guide](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server) to expose the Ollama server to the local network
	

3. **Clone the repository** and navigate to the project directory
4. **Configure the .env file**:
   	Create a ```.env``` file in the root of the project with the following parameters to specify the server settings and model configurations:

	```
 	EMBEDDING_MODEL=sample-embed-model
	CHROMA_ADDRESS=localhost
	CHROMA_PORT=8000
	CHROMA_DB=sample-db
	OLLAMA_ADDRESS=127.0.0.1
	OLLAMA_PORT=11434
 	TEMPERATURE=0.4
 	URL_POPULATE=http://populate:8001/populate
	URL_RETRIEVE=http://retrieve:8002/retrieve
	URL_GENERATE=http://generate:8003/generate
 	```
6. Make the ```compose.sh``` script executable:
	  ```
	  chmod 744 compose.sh
	  ```
7. **Execute** the compose.sh script	
	   ```
	   ./compose.sh
	   ```
   The compose.sh script will orchestrate the setup and startup of all microservices, ensuring the system is ready to handle requests.
