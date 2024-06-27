import faiss
import numpy as np
from document_processing import embed_text

# Define text_chunks as a global variable


def retrieve_chunks(query, text_chunks, index, top_k=5):
    # Embed the query using sentence-transformers
    query_embedding = embed_text(query)

    # Search the FAISS index for similar embeddings
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)

    # Retrieve the corresponding chunks
    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(text_chunks[int(idx)])
    return results