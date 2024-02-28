# Import necessary libraries
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import io
import PyPDF2
import docx

# Function to generate response based on the uploaded file and user query
def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        try:
            # Check file type and extract text accordingly
            if uploaded_file.type == 'application/pdf':
                # Handle PDF file
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                documents = [pdf_reader.pages[i].extract_text() for i in range(len(pdf_reader.pages))]
            elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                # Handle Word document
                doc_reader = docx.Document(io.BytesIO(uploaded_file.read()))
                documents = [para.text for para in doc_reader.paragraphs]
            else:
                # Default to plain text extraction
                documents = [uploaded_file.read().decode()]
            
            # Split documents into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.create_documents(documents)
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            # Create a vector store from documents
            db = Chroma.from_documents(texts, embeddings)
            
            # Create retriever interface
            retriever = db.as_retriever()
            
            # Create and run QA chain
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
            
            # Return the QA response
            return qa.run(query_text)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None

# Streamlit UI setup
# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload widget to accept txt, pdf, and docx files
uploaded_file = st.file_uploader('Upload an article', type=['txt', 'pdf', 'docx'])

# Query text input
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# Input form for API key and query submission
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key:
        with st.spinner('Searching for answers...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)

# Display the response
if len(result):
    st.info(result[0])
