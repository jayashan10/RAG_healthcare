import os
import numpy as np
from document_processing import process_document, embed_text, index_document

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'

# Test document processing
def test_process_document():
    file_path = 'test.txt'
    with open(file_path, 'w') as f:
        f.write('This is a test document. ' * 50)
    chunks = process_document(file_path)
    assert len(chunks) > 0
    print('Document processing test passed.')
    return chunks

# Test embedding generation
def test_embed_text():
    text = 'This is a test document.'
    embedding = embed_text(text)
    assert len(embedding) == 384
    print('Embedding generation test passed.')
    return embedding

# Test document indexing
def test_index_document():
    file_path = 'test.txt'
    chunks = process_document(file_path)
    index_document(file_path)
    print('Document indexing test passed.')
    return chunks

if __name__ == '__main__':
    chunks = test_process_document()
    embedding = test_embed_text()
    indexed_chunks = test_index_document()

