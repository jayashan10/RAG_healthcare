import os
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from tqdm import tqdm
# from multiprocessing import Pool
# from functools import partial
import psutil
import time

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024**2} MB")

# Load the model for embedding generation
model = SentenceTransformer('all-MiniLM-L6-v2')

def process_documents(file_paths):
    # Load documents using llama_index
    documents = SimpleDirectoryReader(input_files=file_paths).load_data()
    
    # Create a more efficient node parser
    node_parser = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    
    # Parse nodes
    nodes = node_parser.get_nodes_from_documents(documents)
    
    # Extract text chunks and metadata
    text_chunks = []
    metadata = []
    for node in nodes:
        text_chunks.append(node.text)
        metadata.append({
            'document': node.metadata.get('file_name', ''),
            'page_number': node.metadata.get('page_label', ''),
            'chunk_id': node.node_id,
        })
    
    return text_chunks, metadata


def embed_text_batch(texts):
    return model.encode(texts)

def embed_text(text):
    return model.encode([text])[0]

def generate_embeddings(text_chunks, batch_size):
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        yield embed_text_batch(batch)

def index_document(file_paths, user_dir):
    print("Indexing docs...")
    
    # Process documents
    text_chunks, metadata = process_documents(file_paths)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(384)  # Dimension of the embeddings
    
    # Batch size for processing
    batch_size = 1024
    
    print_memory_usage()
    # Generate embeddings and add them to the FAISS index
    start_time = time.time()
    for batch_embeddings in generate_embeddings(text_chunks, batch_size):
        print_memory_usage()
        index.add(np.array(batch_embeddings, dtype=np.float32))
    end_time = time.time()

    print_memory_usage()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    # Save the text chunks, metadata, and their embeddings to files in the user-specific directory
    text_chunks_file = os.path.join(user_dir, 'text_chunks_file')
    metadata_file = os.path.join(user_dir, 'metadata_file')
    index_file = os.path.join(user_dir, 'index_file')
    
    with open(text_chunks_file, 'wb') as f:
        pickle.dump(text_chunks, f)
    
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    
    faiss.write_index(index, index_file)
    
    print(f"Indexing complete. Files saved in {user_dir}")
    return text_chunks, metadata, index