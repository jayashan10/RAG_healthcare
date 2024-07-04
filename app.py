import streamlit as st
import os
import uuid
from document_processing import index_document
from query_retrieval import retrieve_chunks
from response_generation import generate_response, generate_response_modal_llama
import pickle
import faiss
import shutil
import streamlit_pills as stp

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

# Function to display compact sources
def display_compact_sources(sources):
    st.write("Sources:", ", ".join(f"[{i+1}]" for i in range(len(sources))))
    for i, source in enumerate(sources):
        with st.expander(f"Source [{i+1}]"):
            st.write(f"**Document:** {source['document']}")
            st.write(f"**Page:** {source['page_number']}")
            st.write(f"**Excerpt:** {source['excerpt']}")


def display_clickable_sources_o(sources, message_index):
    if not sources:
        return
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .sources-container {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px;
        margin-top: 10px;
    }
    .sources-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .stButton > button {
        padding: 2px 8px;
        font-size: 14px;
        border-radius: 4px;
        margin-right: 2px;
    }
    div.row-widget.stHorizontal {
        flex-wrap: wrap;
        gap: 0px;
        justify-content: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a container for the sources
    with st.container():
        st.markdown("<div class='sources-title'>Sources:</div>", unsafe_allow_html=True)
        cols = st.columns(len(sources))
        for i, col in enumerate(cols):
            if col.button(f"[{i+1}]", key=f"source_button_{message_index}_{i}"):
                st.session_state[f'selected_source_{message_index}'] = i

    # Handle button clicks
    if f'selected_source_{message_index}' in st.session_state:
        selected_index = st.session_state[f'selected_source_{message_index}']
        selected_source = sources[selected_index]
        with st.expander(f"Source [{selected_index + 1}] Details", expanded=True):
            st.markdown(f"**Document:** {selected_source['document']}")
            st.markdown(f"**Page:** {selected_source['page_number']}")
            st.markdown(f"**Excerpt:** {selected_source['excerpt']}")


def display_clickable_sources(sources, message_index):
    if not sources:
        return
    
    # Create pills for each source
    pill_labels = [f"{i+1}" for i in range(len(sources))]
    selected_pill = stp.pills("Sources:", pill_labels, key=f"source_pills_{message_index}")

    # Check if a source has been selected and display its details
    if selected_pill:
        selected_index = pill_labels.index(selected_pill)
        selected_source = sources[selected_index]
        with st.expander(f"Source {selected_pill} Details", expanded=True):
            st.markdown(f"**Document:** {selected_source['document']}")
            st.markdown(f"**Page:** {selected_source['page_number']}")
            st.markdown(f"**Excerpt:** {selected_source['excerpt']}")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.viewed_pages = {}
    st.session_state.text_chunks = []
    st.session_state.metadata = []
    st.session_state.index = None
    st.session_state.session_id = str(uuid.uuid4())
    cleanup_user_files()  # Clean up files from previous session

# Set up the Streamlit app
st.set_page_config(page_title="Clinical Trial RAG Assistant", layout="wide")

# Sidebar for document upload
with st.sidebar:
    st.title("Document Upload")
    uploaded_files = st.file_uploader('Upload a clinical trial document', type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

    if uploaded_files:
        user_dir = get_user_dir()
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(user_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)

        if st.button('Process Documents'):
            with st.spinner('Processing documents...'):
                st.session_state.text_chunks, st.session_state.metadata, st.session_state.index = index_document(file_paths, user_dir)
            st.success('Files uploaded and processed successfully!')

    st.title("Model Selection")
    with st.expander("Click here for more information about the models"):
        st.write("""
        **meta-llama/Meta-Llama-3-8B**: This is a large language model trained by Meta. It's good for generating human-like text based on the provided context. It is deployed using Modal Labs

        **OpenAI/GPT-4o**: This is a large language model trained by OpenAI. It's also good for generating human-like text. The responses are generated using the APIs.
        """)
    model = st.radio('Select a model', ('meta-llama/Meta-Llama-3-8B', 'OpenAI/GPT-4o'))

# Main content area
st.title('Clinical Trial Document RAG Assistant')

# Display chat messages from history on app rerun
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            display_clickable_sources(message["sources"], index)
            st.empty()
            
# React to user input
if prompt := st.chat_input("What would you like to know about the clinical trial?"):
    st.chat_message("user").markdown(prompt)

    if st.session_state.index is not None:
        chunks = retrieve_chunks(prompt, st.session_state.text_chunks, st.session_state.metadata, st.session_state.index)
        
        if chunks:
            chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
            if model == 'meta-llama/Meta-Llama-3-8B':
                response = generate_response_modal_llama(chunks, prompt, [msg["content"] for msg in st.session_state.messages])
            else:
                response = generate_response(chunks, prompt, [msg["content"] for msg in st.session_state.messages])
            
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
                if 'sources' in response:
                    display_clickable_sources(response['sources'], len(st.session_state.messages))

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
    cleanup_user_files()
    st.experimental_rerun()