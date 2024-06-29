import streamlit as st
import os
import uuid
from document_processing import index_document
from query_retrieval import retrieve_chunks
from response_generation import generate_response, generate_response_modal_llama
import pickle
import faiss
import shutil

# Function to get user-specific directory
def get_user_dir():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    user_dir = os.path.join('user_data', st.session_state.user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

# Function to clean up user-specific files
def cleanup_user_files():
    user_dir = get_user_dir()
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    print(f"Cleanup completed for user {st.session_state.user_id}")

# Set up the Streamlit app
st.set_page_config(page_title="Clinical Trial RAG Assistant", layout="wide")

# Check if this is a new session
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    cleanup_user_files()  # Clean up files from previous session

# Sidebar for document upload
with st.sidebar:
    st.title("Document Upload")
    uploaded_files = st.file_uploader('Upload a clinical trial document', type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

    file_paths = []
    if uploaded_files:
        user_dir = get_user_dir()
        for uploaded_file in uploaded_files:
            file_path = os.path.join(user_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)

    if st.button('Process Documents') and file_paths:
        with st.spinner('Processing documents...'):
            user_dir = get_user_dir()
            st.session_state.text_chunks, st.session_state.metadata, st.session_state.index = index_document(file_paths, user_dir)
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
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(f"**Document:** {source['document']}")
                    st.markdown(f"**Page:** {source['page_number']}")
                    st.markdown(f"**Excerpt:** {source['excerpt']}")
                    st.markdown("---")


user_dir = get_user_dir()
text_chunks_file = os.path.join(user_dir, 'text_chunks_file')
metadata_file = os.path.join(user_dir, 'metadata_file')
index_file = os.path.join(user_dir, 'index_file')

if os.path.exists(text_chunks_file) and os.path.exists(metadata_file) and os.path.exists(index_file):
    with open(text_chunks_file, 'rb') as f:
        st.session_state.text_chunks = pickle.load(f)
    with open(metadata_file, 'rb') as f:
        st.session_state.metadata = pickle.load(f)
    st.session_state.index = faiss.read_index(index_file)
else:
    st.session_state.text_chunks = []
    st.session_state.metadata = []
    st.session_state.index = None

def get_chat_history_without_sources(messages):
    return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

# React to user input
if prompt := st.chat_input("What would you like to know about the clinical trial?"):
    st.chat_message("user").markdown(prompt)

    if st.session_state.index is not None:
        chunks = retrieve_chunks(prompt, st.session_state.text_chunks, st.session_state.metadata, st.session_state.index)
        
        if chunks:
            chat_history = get_chat_history_without_sources(st.session_state.messages)
            if model == 'meta-llama/Meta-Llama-3-8B':
                response = generate_response_modal_llama(chunks, prompt, [msg["content"] for msg in st.session_state.messages])
            else:
                response = generate_response(chunks, prompt, [msg["content"] for msg in st.session_state.messages])
            
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
                with st.expander("View Sources"):
                    for source in response['sources']:
                        st.markdown(f"**Document:** {source['document']}")
                        st.markdown(f"**Page:** {source['page_number']}")
                        st.markdown(f"**Excerpt:** {source['excerpt']}")
                        st.markdown("---")

            # Add messages to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response['answer'],
                "sources": response['sources']
            })
        else:
            with st.chat_message("assistant"):
                st.markdown("I couldn't find any relevant information in the uploaded documents. Could you please rephrase your question or upload a relevant document?")
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": "I couldn't find any relevant information in the uploaded documents. Could you please rephrase your question or upload a relevant document?"})
    else:
        with st.chat_message("assistant"):
            st.markdown("No documents have been processed yet. Please upload and process some documents first.")
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": "No documents have been processed yet. Please upload and process some documents first."})


# Add a button to clear chat history and user data
if st.button("Clear Chat History"):
    st.session_state.messages = []
    # cleanup_user_files()
    st.experimental_rerun()

# Cleanup on session end
if st.session_state.get('session_ended', False):
    cleanup_user_files()