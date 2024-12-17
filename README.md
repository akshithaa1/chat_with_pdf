# chat_with_pdf
Requirements
Before running the code, ensure you have the following Python libraries installed:

numpy
faiss-cpu
sentence-transformers
transformers
PyMuPDF
ipywidgets
torch
You can install these libraries using the following command:

bash
Copy code
pip install numpy faiss-cpu sentence-transformers transformers PyMuPDF ipywidgets torch
File Structure
pdf_index.faiss: FAISS index file for storing PDF text embeddings.
chunks.npy: Numpy file containing the chunks of extracted PDF text.
script.py: Python script containing the logic for extracting text, chunking, indexing, querying, and generating responses.
Functionality
1. Text Extraction (extract_text_from_pdf)
This function extracts text from a given PDF file using PyMuPDF (fitz). The text from all pages is combined into a single string.

Parameters:
pdf_path: Path to the PDF file.
Output:
Returns the combined text extracted from the PDF.
2. Text Chunking (chunk_text)
This function splits the extracted text into smaller chunks of approximately 500 words each for more efficient processing and embedding.

Parameters:
text: Full text extracted from the PDF.
chunk_size: Number of words per chunk (default is 500).
Output:
Returns a list of text chunks.
3. Embedding and Indexing (embed_and_store)
This function generates embeddings for the text chunks using a pre-trained SentenceTransformer model and stores them in a FAISS index for fast retrieval.

Parameters:
chunks: List of text chunks.
model_name: Name of the embedding model (default is "all-MiniLM-L6-v2").
index_path: Path to save the FAISS index file (default is "pdf_index.faiss").
Output:
Saves the FAISS index to the specified path and the chunks to chunks.npy.
4. Querying the Index (query_index)
This function allows you to query the FAISS index using a user's question. It retrieves the most relevant chunks from the index based on the similarity of their embeddings to the query.

Parameters:
query: The user's question.
model_name: Name of the embedding model (default is "all-MiniLM-L6-v2").
index_path: Path to the FAISS index file (default is "pdf_index.faiss").
Output:
Returns the most relevant text chunks from the index.
5. Response Generation (generate_response_with_huggingface)
This function uses the FLAN-T5-large model from Hugging Face to generate a response based on the retrieved chunks. The query and context are formatted as input, and the model generates a natural language answer.

Parameters:
query: The user's question.
relevant_chunks: The relevant text chunks retrieved from the FAISS index.
Output:
Returns the generated response based on the context and query.
Example Usage
Step 1: Extract Text from PDF
python
Copy code
pdf_text = extract_text_from_pdf("/path/to/your/pdf")
Step 2: Chunk the Extracted Text
python
Copy code
chunks = chunk_text(pdf_text)
Step 3: Embed and Store Chunks
python
Copy code
embed_and_store(chunks)
Step 4: Query the Index and Generate a Response
python
Copy code
query = "Your question here"
relevant_chunks = query_index(query)
response = generate_response_with_huggingface(query, relevant_chunks)
print(response)
Notes
Ensure the FAISS index and text chunks are properly saved and loaded for querying.
The script is intended for querying PDFs and generating answers from their content using machine learning models.
License
This project is licensed under the MIT License - see the LICENSE file for details.
