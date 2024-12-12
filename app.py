import streamlit as st
import os
import uuid
from document_processing import index_document
from query_retrieval import retrieve_chunks
from response_generation import generate_response, generate_response_modal_llama, generate_response_mistral
import pickle
import faiss
import shutil
import streamlit_pills as stp
from prompt_mapping import PROMPT_MAPPING
import streamlit.components.v1 as components
import base64
from streamlit_pdf_viewer import pdf_viewer

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

def handle_file_upload():
    user_dir = get_user_dir()
    uploaded_files = st.file_uploader('Upload a clinical trial document', type=['pdf', 'docx', 'txt'], accept_multiple_files=True, key="file_uploader")
    
    if 'uploaded_file_paths' not in st.session_state:
        st.session_state.uploaded_file_paths = []

    current_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(user_dir, uploaded_file.name)
        current_files.append(file_path)
        if file_path not in st.session_state.uploaded_file_paths:
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_file_paths.append(file_path)

    # Check for files that were removed from the uploader
    for file_path in st.session_state.uploaded_file_paths[:]:
        if file_path not in current_files:
            if delete_file(file_path):
                st.session_state.uploaded_file_paths.remove(file_path)
                st.success(f"File {os.path.basename(file_path)} has been deleted.")

    return st.session_state.uploaded_file_paths


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

def get_preconfigured_prompts():
    return list(PROMPT_MAPPING.keys())

# Function to display compact sources
def display_compact_sources(sources):
    st.write("Sources:", ", ".join(f"[{i+1}]" for i in range(len(sources))))
    for i, source in enumerate(sources):
        with st.expander(f"Source [{i+1}]"):
            st.write(f"**Document:** {source['document']}")
            st.write(f"**Page:** {source['page_number']}")
            st.write(f"**Excerpt:** {source['excerpt']}")

def display_pdf_viewer(file_path):
    try:
        # Try to display the PDF directly using the file path
        pdf_viewer(file_path)
    except Exception as e:
        # If direct display fails, fall back to base64 encoding
        st.warning(f"Direct PDF display failed. Falling back to base64 encoding. Error: {e}")
        try:
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_viewer(base64_pdf)
        except Exception as e:
            st.error(f"Failed to display PDF. Error: {e}")


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

def format_model_label(model):
    if model == 'meta-llama/Meta-Llama-3-8B':
        return 'meta-llama/Meta-Llama-3-8B'
    elif model == 'Mistral-7B-Instruct':
        return 'Mistral-7B-Instruct (slow)'
    return model



if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.viewed_pages = {}
    st.session_state.text_chunks = []
    st.session_state.metadata = []
    st.session_state.index = None
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.pill_key = 0
    st.session_state.current_page = "main"
    cleanup_user_files() # Clean up files from previous session

# Set up the Streamlit app
st.set_page_config(page_title="Clinical Trial Protocol Assistant", layout="wide")
# Sidebar for document upload
with st.sidebar:
    st.title("Document Upload")
    use_sample_file = st.checkbox('Use sample clinical trial protocol')
    
    if use_sample_file:
        sample_file_path = 'clinical_trial_protocol.pdf'
        file_dir = './sample_data'
        # Ensure the full path is used
        full_sample_path = os.path.join(file_dir, sample_file_path)
        file_paths = [full_sample_path]
        st.session_state.sample_file_selected = True
        if st.button("View Sample File"):
            st.session_state.current_page = "sample_file_view"
    else:
        st.session_state.sample_file_selected = False
        file_paths = handle_file_upload()

    if st.button('Process Documents'):
        with st.spinner('Processing documents...'):
            print("The file paths are: ", file_paths)
            # Use get_user_dir() only for uploaded files, not for sample files
            base_dir = file_dir if use_sample_file else get_user_dir()
            st.session_state.text_chunks, st.session_state.metadata, st.session_state.index = index_document(file_paths, base_dir)
        st.success('Files uploaded and processed successfully!')

    st.title("Model Selection")
    # **OpenAI/GPT-4o**: This is a large language model trained by OpenAI. It's also good for generating human-like text. The responses are generated using the APIs.
    with st.expander("Click here for more information about the models"):
        st.write("""
        **meta-llama/Meta-Llama-3-8B**: This is a large language model trained by Meta. It's good for generating human-like text based on the provided context. It is deployed using Modal Labs

        **Mistral-7B-Instruct**: A powerful open-source language model from Mistral AI, 
        offering strong performance for instruction following and general text generation.
        """)
        
    model = st.radio(
    'Select a model', 
    ('meta-llama/Meta-Llama-3-8B', 'Mistral-7B-Instruct'), 
    format_func=format_model_label
)

# Main content area
if st.session_state.current_page == "main":
    st.title('Clinical Trial Protocol Assistant')
    
    # Display preconfigured prompts as pills
    st.write("Common Clinical Trial Questions:")
    preconfigured_prompts = get_preconfigured_prompts()
    selected_prompt = stp.pills("", preconfigured_prompts, key=f"pill_{st.session_state.pill_key}", index=None)

    # Display chat messages from history on app rerun
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                display_clickable_sources(message["sources"], index)
                st.empty()

    # Display chat input box for typing a custom question
    custom_prompt = st.chat_input("What would you like to know about the clinical trial?", key="user_input")

    # React to user input
    if custom_prompt:
        frontend_prompt = custom_prompt
        backend_prompt = custom_prompt
    elif selected_prompt:
        frontend_prompt = selected_prompt
        backend_prompt = PROMPT_MAPPING[selected_prompt]
    else:
        frontend_prompt = None
        backend_prompt = None
                
    # React to user input
    if frontend_prompt:
        st.chat_message("user").markdown(frontend_prompt)

        if st.session_state.index is not None:
            chunks = retrieve_chunks(backend_prompt, st.session_state.text_chunks, st.session_state.metadata, st.session_state.index)
            
            if chunks:
                chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
                if model == 'meta-llama/Meta-Llama-3-8B':
                    response = generate_response_modal_llama(chunks, backend_prompt, [msg["content"] for msg in st.session_state.messages])
                elif model == 'Mistral-7B-Instruct':
                    response = generate_response_mistral(chunks, backend_prompt, [msg["content"] for msg in st.session_state.messages])
                else:
                    response = generate_response(chunks, backend_prompt, [msg["content"] for msg in st.session_state.messages])
                
                with st.chat_message("assistant"):
                    st.markdown(response['answer'])
                    if 'sources' in response:
                        display_clickable_sources(response['sources'], len(st.session_state.messages))

                # Add messages to chat history
                st.session_state.messages.append({"role": "user", "content": frontend_prompt})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response['answer'],
                    "sources": response['sources']
                })
            else:
                with st.chat_message("assistant"):
                    st.markdown("I couldn't find any relevant information in the uploaded documents. Could you please rephrase your question or upload a relevant document?")
                st.session_state.messages.append({"role": "user", "content": frontend_prompt})
                st.session_state.messages.append({"role": "assistant", "content": "I couldn't find any relevant information in the uploaded documents. Could you please rephrase your question or upload a relevant document?"})
        else:
            with st.chat_message("assistant"):
                st.markdown("No documents have been processed yet. Please upload and process some documents first.")
            st.session_state.messages.append({"role": "user", "content": frontend_prompt})
            st.session_state.messages.append({"role": "assistant", "content": "No documents have been processed yet. Please upload and process some documents first."})

        # Reset pill selection after processing
        st.session_state.pill_key += 1
        st.rerun()

    # Add a button to clear chat history and user data
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.pill_key += 1  # Increment pill key to reset pill component
        # cleanup_user_files()
        st.rerun()

elif st.session_state.current_page == "sample_file_view":
    st.title("Sample Clinical Trial Protocol")
    if st.button("Back to Chat"):
        st.session_state.current_page = "main"
        st.rerun()
    sample_file_path = os.path.join('sample_data', 'clinical_trial_protocol.pdf')
    if os.path.exists(sample_file_path):
        st.write("PDF Viewer:")
        display_pdf_viewer(sample_file_path)
    else:
        st.error(f"Sample file not found at {sample_file_path}. Please make sure the file exists.")
    
    