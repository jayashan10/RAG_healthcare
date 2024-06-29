import os
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SimpleNodeParser, TokenTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load the model for embedding generation
model = SentenceTransformer('all-MiniLM-L6-v2')

def process_documents(directory_path):
    # Load documents using llama_index
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    # Create a node parser
    node_parser = SimpleNodeParser(
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

def embed_text(text):
    return model.encode([text])[0]

def index_document(file_paths, user_dir):
    print("Indexing docs...")
    
    # Process documents
    text_chunks, metadata = process_documents(user_dir)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(384)  # Dimension of the embeddings
    
    for chunk in text_chunks:
        embedding = model.encode(chunk)
        index.add(np.array([embedding], dtype=np.float32))
    
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