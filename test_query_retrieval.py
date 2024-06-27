import os
import numpy as np
from document_processing import process_document, embed_text, index_document
from query_retrieval import retrieve_chunks

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'

# Test document processing and indexing
def test_index_document():
    file_path = 'test.txt'
    with open(file_path, 'w') as f:
        f.write('This is a test document. ' * 50)
    index_document(file_path)
    print('Document indexing test passed.')

# Test query retrieval
def test_retrieve_chunks():
    query = 'test'
    chunks = retrieve_chunks(query)
    assert len(chunks) > 0
    print('Query retrieval test passed.')
    return chunks

if __name__ == '__main__':
    test_index_document()
    retrieved_chunks = test_retrieve_chunks()

