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

2. **Clone the repository** and navigate to the project directory
3. Make the ```compose.sh``` script executable:
	  ```
	  chmod 744 compose.sh
	  ```
5. 
