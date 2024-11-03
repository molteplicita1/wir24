# Web Information Retrieval - Course Notes Retrieval System

This project is a Retrieval-Augmented Generation (RAG) system designed to retrieve and process course notes from the undergraduate Computer Engineering program. 
The system allows users to perform queries over the collection of notes using different models and strategies.

## Description

The system integrates document retrieval and query processing. It extracts text from course notes (PDF files), processes the text into chunks, stores it in a vector database and then uses LLM models to answer queries.

Tokenization can be handled by either a simple regex-based approach or a more sophisticated method using pre-trained models.

Additionally, the system requires HTTP calls to an Ollama server that runs the LLMs. The server's address must be specified in the ```.env``` file under the ```OLLAMA_ADDRESS``` and ```OLLAMA_PORT``` fields for the system to function correctly.


## Project Structure  

-  **client.py**: executes a single query over the course notes using the model specified in the `.env` file. The result is saved in an `output.md` file, which contains:

	  - the query
	
	  - the retrieved documents
	
	  - the model's response

  	The query must be specified directly within the `client.py` file before running the script.
  	The query must be specified directly within the `client.py` file before running the script.

-  **pipe_models.py**: compares various models, tokenizers, and embeddings by processing multiple queries. The results are stored in an `output.md` file, which includes:

	- the model used

	- the embedding technique

	- the tokenizer used

	- the query

	- the retrieved documents

	- the model's response

  

-  **functions.py**: contains utility functions such as:

	-  `extract_text_from_pdf_files()`: extracts text content from PDFs located in a specified directory

	-  `chunk_splitter()`: splits the text into chunks to improve tokenization and model processing

	-  `pipeline()`: executes the retrieval and query processing pipeline using the provided model and query

  

-  **tokenizer.py**: defines strategies for text tokenization:

	-  `SimpleTokenizer`: tokenizes text using a basic regex-based approach

	-  `PretrainedTokenizer`: tokenizes text using a pre-trained model from the Hugging Face `transformers` library

	-  `TokenizerFactory`: a factory class for creating tokenizer objects, with or without a pre-trained model


-  **queries.txt**: a list of sample queries to be used for retrieval tests

-  **embeddings.txt**: a list of embeddings

-  **models.txt**: a list of models

-  **requirements.txt**: lists the Python dependencies required to run the project

  

## How to Run

1. **Install Ollama**
	
	Download and install Ollama from [here](https://ollama.com/download).

	After the installation, follow this [guide](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server
) to expose the Ollama server to the local network 
2. **Install Chroma with Docker**
	
	Run the following command to install ChromaDB:

	```
	docker run -d -p 8000:8000 -v chroma-data:/chromadb/data -e ALLOW_RESET=TRUE --name chroma chromadb/chroma:0.5.15
	```

3. **Configure the .env file**
   
   	Create a ```.env``` file in the root of the project with the following parameters to specify the server settings and model configurations:

	```
 	EMBEDDING_MODEL=sample-embed-model
	TOKENIZER=sample-tokenizer
	CHROMA_ADDRESS=localhost
	CHROMA_PORT=8000
	CHROMA_DB=sample-db
	DATA_PATH=sample-data-path
	MODEL=sample-model
	OLLAMA_ADDRESS=127.0.0.1
	OLLAMA_PORT=11434
 	TEMPERATURE=0.4
 	```

4.	**Add documents**

	Create a ```data``` directory and add the PDF documents which wants include in the retrieve system 

5.  **Install dependencies**

	Run the following command to install all required Python libraries:

	```
	pip install -r requirements.txt
	```

6.  **Execute a single query**

	Update the client.py file with the desired query, then run:

	```
	python client.py
	```

	The result will be saved in the output.md file, which will include the query, the retrieved documents, and the model's response.

    To reset the database, run the following command:

    ```
	python client.py --reset
	```

7.  **Compare multiple models**:

	Use the pipe_models.py script to compare different models, embeddings, and tokenizers. 		
    Update the ```.txt``` files with the models, techniques and queries you want to test, then run:

	```
	python pipe_models.py --reset
	```

	The results will be saved in the output.md file, detailing the model, embedding, tokenizer, query, retrieved documents, and model's response.

8. **Compare multiple temperatures**:
   
   Use the pipe_queries.py script to compare various temperature settings on a single model.
Update the list of temperatures in the file, then run:
   ```
	python pipe_queries.py
   ```
   The results will be saved in separate files for each selected temperature, with each file named in the format ```output_{temperature}.md```.
   


