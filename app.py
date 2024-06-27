import streamlit as st
import os
from document_processing import index_document
from query_retrieval import retrieve_chunks
from response_generation import generate_response, generate_response_modal_llama
import pickle
import faiss
import shutil

# Function to clean up files
def cleanup():
    # Remove uploaded files
    if os.path.exists('uploads'):
        shutil.rmtree('uploads')
    
    # Remove index file
    if os.path.exists('index_file'):
        os.remove('index_file')
    
    # Remove text chunks file
    if os.path.exists('text_chunks_file'):
        os.remove('text_chunks_file')
    
    print("Cleanup completed")

# Set up the Streamlit app
st.set_page_config(page_title="Clinical Trial RAG Assistant", layout="wide")

# Check if this is a new session
if 'session_id' not in st.session_state:
    # Generate a new session ID
    st.session_state.session_id = os.urandom(16).hex()
    # Perform cleanup from any previous session
    cleanup()

# Sidebar for document upload
with st.sidebar:
    st.title("Document Upload")
    uploaded_files = st.file_uploader('Upload a clinical trial document', type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

    file_paths = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Create 'uploads' directory if it doesn't exist
            os.makedirs('uploads', exist_ok=True)
            # Save the uploaded file to the 'uploads' directory
            file_path = os.path.join('uploads', uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)

    if st.button('Process Documents'):
        with st.spinner('Processing documents...'):
            # Index the uploaded document
            index_document(file_paths)
        st.success('Files uploaded and processed successfully!')

    st.title("Model Selection")
    with st.expander("Click here for more information about the models"):
        st.write("""
        **meta-llama/Meta-Llama-3-8B**: This is a large language model trained by Meta. It's good for generating human-like text based on the provided context.

        **OpenAI-GPT4**: This is a large language model trained by OpenAI. It's also good for generating human-like text, but it might have different strengths and weaknesses compared to Meta-Llama-3-8B.
        """)
    model = st.radio('Select a model', ('meta-llama/Meta-Llama-3-8B', 'OpenAI-GPT4'))

# Main content area
st.title('Clinical Trial Document RAG Assistant')

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if os.path.exists('text_chunks_file'):
    with open('text_chunks_file', 'rb') as f:
        st.session_state.text_chunks = pickle.load(f)
    st.session_state.index = faiss.read_index('index_file')
else:
    st.session_state.text_chunks = []
    
# React to user input
if prompt := st.chat_input("What would you like to know about the clinical trial?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Generate a response
    chunks = retrieve_chunks(prompt, st.session_state.text_chunks, st.session_state.index)
    if chunks:
        if model == 'meta-llama/Meta-Llama-3-8B':
            response = generate_response_modal_llama(chunks, prompt, [msg["content"] for msg in st.session_state.messages])
        else:
            response = generate_response(chunks, prompt, [msg["content"] for msg in st.session_state.messages])
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            st.markdown("I couldn't find any relevant information in the uploaded documents. Could you please rephrase your question or upload a relevant document?")
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": "I couldn't find any relevant information in the uploaded documents. Could you please rephrase your question or upload a relevant document?"})

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()