import faiss
import numpy as np
from document_processing import embed_text
from langsmith import Client
from langsmith.run_helpers import traceable

# Initialize LangSmith client
client = Client()

@traceable(run_type="retriever")
def retrieve_chunks(query, text_chunks, metadata, index, top_k=10):
    # Embed the query using sentence-transformers
    query_embedding = embed_text(query)

    # Search the FAISS index for similar embeddings
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)

    # Retrieve the corresponding chunks and metadata
    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append({
                'text': text_chunks[int(idx)],
                'metadata': metadata[int(idx)],
                # 'score': float(distances[0][int(idx)])
            })
    return results