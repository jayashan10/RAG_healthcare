import os
import fitz  # PyMuPDF
import docx
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load the model for embedding generation
model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.IndexFlatL2(384)  # Dimension of the embeddings
text_chunks = []

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def chunk_text(text, chunk_size=500):
    # Split text into chunks of specified size
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_pdf(file_path):
    # Extract text from PDF
    doc = fitz.open(file_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return chunk_text(text)

def process_docx(file_path):
    # Extract text from DOCX
    doc = docx.Document(file_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text
    return chunk_text(text)

def process_txt(file_path):
    # Extract text from TXT
    with open(file_path, 'r') as file:
        text = file.read()
    return chunk_text(text)

def process_document(file_path):
    # Determine file type and process accordingly
    if file_path.endswith('.pdf'):
        return process_pdf(file_path)
    elif file_path.endswith('.docx'):
        return process_docx(file_path)
    elif file_path.endswith('.txt'):
        return process_txt(file_path)
    else:
        raise ValueError('Unsupported file type')

def embed_text(text):
    # Generate embeddings using sentence-transformers
    return model.encode([text])[0]

def index_document(file_paths):
    print("indexing docs...........")
    for file_path in file_paths:
        # Process the document and index the chunks
        chunks = process_document(file_path)
        for chunk in chunks:
            text_chunks.append(chunk)
            embedding = model.encode(chunk)
            index.add(np.array([embedding], dtype=np.float32))
    
    # Save the text chunks and their embeddings to files
    with open('text_chunks_file', 'wb') as f:
        pickle.dump(text_chunks, f)
    faiss.write_index(index, 'index_file')



