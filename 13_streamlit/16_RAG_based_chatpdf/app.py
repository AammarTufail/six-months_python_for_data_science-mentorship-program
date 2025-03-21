import streamlit as st
import openai
import PyPDF2
import docx2txt
import os
import tempfile
from io import StringIO
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import textwrap

# -----------------------------
# 1. Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Document Chatbot - RAG App",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# 2. Custom CSS for Enhanced UI
# -----------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
    }
    .chat-container {
        max-width: 800px;
        margin: auto;
    }
    .user-message {
        background-color: #e1ffc7;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        word-wrap: break-word;
    }
    .assistant-message {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        word-wrap: break-word;
    }
    .message {
        display: flex;
        align-items: flex-start;
    }
    .message .avatar {
        margin-right: 10px;
    }
    .sidebar .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# 3. Session State Variables
# -----------------------------
if 'messages' not in st.session_state:
    # Holds the entire conversation history
    st.session_state.messages = []
if 'edit_index' not in st.session_state:
    st.session_state.edit_index = None
if 'thread_name' not in st.session_state:
    st.session_state.thread_name = 'Default'
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'document_texts' not in st.session_state:
    st.session_state.document_texts = []
if 'uploaded_files_info' not in st.session_state:
    st.session_state.uploaded_files_info = None
if 'doc_chunks' not in st.session_state:
    st.session_state.doc_chunks = []
if 'chunk_embeddings' not in st.session_state:
    st.session_state.chunk_embeddings = None

# -----------------------------
# 4. Helper Functions
# -----------------------------

def get_ai_response(messages, api_key, model, provider):
    """
    Queries the Perplexity or OpenAI API with the conversation
    messages and returns the assistant's response.
    """
    openai.api_key = api_key
    if provider == 'Perplexity':
        openai.api_base = "https://api.perplexity.ai"
    else:
        openai.api_base = "https://api.openai.com/v1"

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=500,    # Limit the response length
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"


def process_uploaded_files(uploaded_files):
    """
    Extract text from each uploaded file (PDF, DOCX, TXT)
    and return a list of texts.
    """
    document_texts = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            # Read PDF file
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            document_texts.append(text)

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Read DOCX file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(uploaded_file.read())
                tmp.seek(0)
                text = docx2txt.process(tmp.name)
                document_texts.append(text)
            os.unlink(tmp.name)

        elif uploaded_file.type == "text/plain":
            # Read TXT file
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
            document_texts.append(text)

    return document_texts


def split_text(text, chunk_size=500, overlap=50):
    """
    Splits the text into chunks of specified size with overlap.
    This helps in more precise retrieval of relevant text segments.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def create_vector_store(chunks):
    """
    Creates a FAISS vector store for all the chunks, using sentence-transformers.
    Returns the index, the embedding model, and the embeddings array.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, model, embeddings


def retrieve_chunks(query, index, model, chunks, k=5):
    """
    Retrieves the top-k relevant chunks for the query from the FAISS index.
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    return retrieved_chunks

# -----------------------------
# 5. Sidebar - Settings & Document Upload
# -----------------------------
with st.sidebar:
    st.title("Settings")

    # Choose API provider
    provider = st.radio("Select AI Provider", options=['Perplexity', 'OpenAI'])

    # Enter API Key
    api_key = st.text_input(f"Enter {provider} API Key", type="password")

    # Model Selection
    st.subheader("Select Model")
    if provider == 'Perplexity':
        models = [
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-huge-128k-online",
            "llama-3.1-sonar-small-128k-chat",
            "llama-3.1-sonar-large-128k-chat",
            "llama-3.1-8b-instruct",
            "llama-3.1-70b-instruct"
        ]
    else:
        models = [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-instruct",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-turbo",
        ]
    model = st.selectbox("Choose a model", models)

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        accept_multiple_files=True
    )

    # Process uploaded files if they changed
    if uploaded_files:
        current_files_info = [(file.name, file.size) for file in uploaded_files]
        if st.session_state.uploaded_files_info != current_files_info:
            with st.spinner("Processing documents..."):
                # Extract text from uploaded files
                st.session_state.document_texts = process_uploaded_files(uploaded_files)
                # Split documents into chunks
                st.session_state.doc_chunks = []
                for doc_text in st.session_state.document_texts:
                    chunks = split_text(doc_text)
                    st.session_state.doc_chunks.extend(chunks)
                # Create vector store
                st.session_state.vector_store, st.session_state.embeddings_model, st.session_state.chunk_embeddings = create_vector_store(st.session_state.doc_chunks)
                st.session_state.uploaded_files_info = current_files_info
            st.success("Documents processed and vector store created.")

    # Thread saving and loading
    st.subheader("Threads")
    if 'threads' not in st.session_state:
        st.session_state.threads = {}
    thread_list = list(st.session_state.threads.keys())
    if thread_list:
        selected_thread = st.selectbox("Load a thread", options=thread_list)
        if st.button("Load Thread"):
            st.session_state.messages = st.session_state.threads[selected_thread]['messages']
            st.session_state.thread_name = selected_thread
            st.experimental_rerun()
    thread_name_input = st.text_input(
        "Save current thread as",
        value=st.session_state.thread_name
    )
    if st.button("Save Thread"):
        st.session_state.thread_name = thread_name_input
        st.session_state.threads[thread_name_input] = {
            'messages': st.session_state.messages.copy()
        }
        st.success(f"Thread '{thread_name_input}' saved.")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.edit_index = None
        st.session_state.thread_name = 'Default'
        st.experimental_rerun()

# -----------------------------
# 6. Main Chat Interface
# -----------------------------
st.title("📄 Document Chatbot - RAG App")
st.write("Chat with your documents using AI.")

# Display Chat History
with st.container():
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(
                f'<div class="message user-message"><strong>You:</strong> {message["content"]}</div>',
                unsafe_allow_html=True
            )
        elif message["role"] == "assistant":
            st.markdown(
                f'<div class="message assistant-message"><strong>Assistant:</strong> {message["content"]}</div>',
                unsafe_allow_html=True
            )

# Chat Input
if st.session_state.edit_index is None:
    prompt = st.chat_input("Type your message here and press Enter")
    if prompt:
        if not api_key:
            st.error(f"Please enter your {provider} API key in the sidebar.")
        else:
            # 1) Add user message to the conversation
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            # 2) Retrieve relevant chunks from vector store
            if st.session_state.vector_store:
                context_chunks = retrieve_chunks(
                    prompt,
                    st.session_state.vector_store,
                    st.session_state.embeddings_model,
                    st.session_state.doc_chunks,
                    k=5  # Number of chunks to retrieve
                )
                context = "\n\n".join(context_chunks)
                # Limit context length for safety
                max_context_length = 1500
                if len(context) > max_context_length:
                    context = context[:max_context_length]
            else:
                context = ""

            # 3) Prepare messages for the AI call
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that answers questions "
                        "based on the provided context. Use the context to "
                        "provide accurate answers. Do not mention the context "
                        "in your responses."
                    )
                }
            ] + st.session_state.messages

            # Insert context as a system message
            if context:
                messages.insert(
                    1, 
                    {
                        "role": "system",
                        "content": f"Context:\n{context}"
                    }
                )

            # 4) Get Assistant Response
            with st.spinner("Thinking..."):
                response = get_ai_response(
                    messages=messages,
                    api_key=api_key,
                    model=model,
                    provider=provider
                )
                # 5) Save Assistant message to conversation
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            # Rerun to update UI
            st.experimental_rerun()
