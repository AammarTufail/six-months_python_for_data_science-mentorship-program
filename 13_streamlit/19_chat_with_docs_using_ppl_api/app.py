import os
import pickle
import tempfile
import textwrap
from io import StringIO
from typing import List

import fitz  # PyMuPDF - faster PDF text extraction
import docx2txt
import joblib
import numpy as np
import streamlit as st
import faiss
import torch
from sentence_transformers import SentenceTransformer
import openai

# -----------------------------
# 1. Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fast RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# 2. Check MPS (Metal) or CPU
# -----------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
    st.write("✅ Using MPS on Apple Silicon for faster embeddings.")
else:
    DEVICE = "cpu"
    st.write("MPS not available; using CPU.")

# -----------------------------
# 3. Custom CSS for UI
# -----------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
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
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# 4. Session State
# -----------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'doc_chunks' not in st.session_state:
    st.session_state.doc_chunks = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'chunk_embeddings' not in st.session_state:
    st.session_state.chunk_embeddings = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None

# -----------------------------
# 5. RAG Helper Functions
# -----------------------------

def extract_text_pymupdf(file_bytes) -> str:
    """Use PyMuPDF (fitz) for fast PDF extraction."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = []
    for page in doc:
        texts.append(page.get_text())
    return "\n".join(texts)

def process_uploaded_file(uploaded_file) -> str:
    """Process a single file: PDF, DOCX, TXT."""
    if uploaded_file.type == "application/pdf":
        # Use PyMuPDF
        return extract_text_pymupdf(uploaded_file.read())

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # DOCX
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp.seek(0)
            text = docx2txt.process(tmp.name)
        os.unlink(tmp.name)
        return text

    elif uploaded_file.type == "text/plain":
        # TXT
        return uploaded_file.read().decode("utf-8")

    return ""

def chunk_text(text: str, chunk_size=1000, overlap=100) -> List[str]:
    """
    Split text into large chunks, to reduce total embeddings and speed up retrieval.
    Overlap helps preserve context across chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_chunks_parallel(chunks: List[str], model: SentenceTransformer, n_jobs=4) -> np.ndarray:
    """
    Use Joblib to parallelize embedding for faster CPU usage.
    On MPS, multi-process might not help as much, but it can still help CPU-bound tasks.
    """
    def embed_single_chunk(text_chunk):
        return model.encode(text_chunk, device=DEVICE)

    embeddings = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
        joblib.delayed(embed_single_chunk)(chunk) for chunk in chunks
    )
    return np.array(embeddings)

def create_or_load_index(cache_file: str, chunks: List[str]) -> faiss.IndexFlatL2:
    """
    Demonstrates caching & incremental indexing:
    - If 'cache_file' exists, load the FAISS index from disk.
    - Otherwise, embed & create new index, then save to disk.
    """
    if os.path.exists(cache_file):
        st.write("Loading cached FAISS index...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        st.session_state.faiss_index = data['index']
        st.session_state.chunk_embeddings = data['embeddings']
    else:
        st.write("No cache found. Building new FAISS index...")

        # Create a SentenceTransformer on MPS or CPU
        model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
        # Parallel embedding
        st.write("Embedding chunks in parallel. Please wait...")
        embeddings = embed_chunks_parallel(chunks, model)

        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        st.session_state.faiss_index = index
        st.session_state.chunk_embeddings = embeddings

        # Save to cache for next time
        with open(cache_file, "wb") as f:
            pickle.dump({'index': index, 'embeddings': embeddings}, f)

    return st.session_state.faiss_index

def retrieve_top_k(query: str, index: faiss.IndexFlatL2, chunks: List[str], model: SentenceTransformer, k=5) -> List[str]:
    """Retrieve top-k chunks from FAISS for the given query."""
    # Encode the query on the same device
    query_embedding = model.encode([query], device=DEVICE)
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    return retrieved_chunks

# OpenAI or Perplexity
def query_llm(messages, api_key, model, provider):
    openai.api_key = api_key
    if provider == 'Perplexity':
        openai.api_base = "https://api.perplexity.ai"
    else:
        openai.api_base = "https://api.openai.com/v1"

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=400,  # smaller limit for speed
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# 6. SIDEBAR: Settings, Upload, Cache
# -----------------------------
st.sidebar.title("Settings")

provider = st.sidebar.radio("Select AI Provider", ['Perplexity', 'OpenAI'])
api_key = st.sidebar.text_input(f"Enter {provider} API Key", type="password")

st.sidebar.subheader("Select Model")
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
        "gpt-4-turbo"
    ]

model = st.sidebar.selectbox("Choose a model", models)

st.sidebar.subheader("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", accept_multiple_files=True)

cache_path = "faiss_index.pkl"  # local cache file for embeddings

# If user uploads new files
if uploaded_files:
    # Gather names & sizes for comparison
    new_files_info = [(file.name, file.size) for file in uploaded_files]
    if st.session_state.uploaded_files != new_files_info:
        # Process
        st.session_state.doc_chunks.clear()
        text_blocks = []
        for uf in uploaded_files:
            text = process_uploaded_file(uf)
            # chunk the text
            chunks = chunk_text(text, chunk_size=1000, overlap=100)
            text_blocks.extend(chunks)
        # Store chunks
        st.session_state.doc_chunks = text_blocks
        st.session_state.uploaded_files = new_files_info

        # Build or rebuild the index (or load from cache)
        st.session_state.faiss_index = create_or_load_index(
            cache_file=cache_path,
            chunks=st.session_state.doc_chunks
        )
else:
    st.write("Upload PDF/DOCX/TXT to build or load a cached index.")

# Clear Chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# -----------------------------
# 7. MAIN APP: Chat UI
# -----------------------------
st.title("Fast RAG Chatbot with MPS & Parallel Embedding")

# Display conversation
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f"<div class='user-message'><strong>You:</strong> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-message'><strong>Assistant:</strong> {msg['content']}</div>", unsafe_allow_html=True)

prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    if not api_key:
        st.error(f"Please enter your {provider} API key in the sidebar.")
    else:
        # 1) Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2) Retrieve top-k chunks
        if st.session_state.faiss_index and len(st.session_state.doc_chunks) > 0:
            # we need a SentenceTransformer for retrieving, but we can reuse the same model
            embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
            top_chunks = retrieve_top_k(prompt, st.session_state.faiss_index, st.session_state.doc_chunks, embed_model)
            context = "\n\n".join(top_chunks)
            if len(context) > 1500:
                context = context[:1500]
        else:
            context = ""

        # 3) Prepare final messages
        messages_to_send = [
            {
                "role": "system",
                "content": "You are an AI assistant that answers questions using the provided context. Do not reveal the context explicitly."
            }
        ] + st.session_state.messages

        if context:
            messages_to_send.insert(
                1,
                {
                    "role": "system",
                    "content": f"Context:\n{context}"
                }
            )

        # 4) Query the LLM
        with st.spinner("Generating answer..."):
            llm_response = query_llm(
                messages=messages_to_send,
                api_key=api_key,
                model=model,
                provider=provider
            )

        # 5) Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": llm_response})

        # Rerun so the chat is updated
        st.experimental_rerun()
