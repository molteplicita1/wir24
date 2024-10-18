# Web Information Retrieval - Course Notes Retrieval System

This project is a Retrieval-Augmented Generation (RAG) system designed to retrieve and process course notes from the undergraduate Computer Engineering program. 
The system allows users to perform queries over the collection of notes using different models and strategies.


## Project Structure  

-  **client.py**: Executes a single query over the course notes using the model specified in the `.env` file. The result is saved in an `output.md` file, which contains:

	  - The query,
	
	  - The retrieved documents,
	
	  - The model's response.

  	The query must be specified directly within the `client.py` file before running the script.

-  **pipe.py**: Compares various models, tokenizers, and embeddings by processing multiple queries. The results are stored in an `output.md` file, which includes:

	- The model used,

	- The embedding technique,

	- The tokenizer used,

	- The query,

	- The retrieved documents,

	- The model's response.

  

-  **functions.py**: Contains utility functions such as:

	-  `extract_text_from_pdf_files()`: Extracts text content from PDFs located in a specified directory.

	-  `chunk_splitter()`: Splits the text into chunks to improve tokenization and model processing.

	-  `pipeline()`: Executes the retrieval and query processing pipeline using the provided model and query.

  

-  **tokenizer.py**: Defines strategies for text tokenization:

	-  `SimpleTokenizer`: Tokenizes text using a basic regex-based approach.

	-  `PretrainedTokenizer`: Tokenizes text using a pre-trained model from the Hugging Face `transformers` library.

	-  `TokenizerFactory`: A factory class for creating tokenizer objects, with or without a pre-trained model.


-  **queries.txt**: A list of sample queries to be used for retrieval tests.

-  **embeddings.txt**: A list of embeddings.

-  **models.txt**: A list of models.

-  **requirements.txt**: Lists the Python dependencies required to run the project.

  

## How to Run

1.  **Install dependencies**:

	Run the following command to install all required Python libraries:

	```
	pip install -r requirements.txt
	```

2.  **Execute a single query**:

	Update the client.py file with the desired query, then run:

	```
	python client.py
	```

	The result will be saved in the output.md file, which will include the query, the retrieved documents, and the model's response.

3.  **Compare multiple models**:

	Use the pipe.py script to compare different models, embeddings, and tokenizers. 		
  Update the script with the models and techniques you want to test, then run:

	```
	python pipe.py
	```

	The results will be saved in the output.md file, detailing the model, embedding, tokenizer, query, retrieved documents, and model's response.

## Description

This project integrates document retrieval and query processing.

It extracts text from course notes (PDF files), processes the text into chunks, and then uses machine learning models to answer queries.

Tokenization can be handled by either a simple regex-based approach or a more sophisticated method using pre-trained models.
