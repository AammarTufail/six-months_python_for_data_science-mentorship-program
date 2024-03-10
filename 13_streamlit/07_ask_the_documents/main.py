# Importing necessary libraries
import streamlit as st
import PyPDF2
import io
import openai
import docx2txt
import pyperclip
import os  # Added the OS module to work with file directories

# Setting up OpenAI API key
openai.api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# Defining a function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

# Function to list PDF files in a directory
def list_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, filename))
    return pdf_files

# Updating function to generate questions from text using OpenAI's updated API
def get_questions_from_gpt(text):
    prompt = text[:4096]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5, 
        max_tokens=30
    )
    return response['choices'][0]['message']['content'].strip()

# Updating function to generate answers to a question using OpenAI's updated API
def get_answers_from_gpt(text, question):
    prompt = text[:4096] + "\nQuestion: " + question + "\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6, 
        max_tokens=2000
    )
    return response['choices'][0]['message']['content'].strip()

# Defining the main function of the Streamlit app
def main():
    st.title("Ask Questions From PDF Documents in Folder")
    
    # Get the folder containing PDF files using folder input
    pdf_folder = st.text_input("Enter the folder path containing PDF files:")
    
    if pdf_folder and os.path.isdir(pdf_folder):
        pdf_files = list_pdf_files(pdf_folder)
        
        if not pdf_files:
            st.warning("No PDF files found in the specified folder.")
        else:
            st.info(f"Number of PDF files found: {len(pdf_files)}")
            
            # Select PDF file
            selected_pdf = st.selectbox("Select a PDF file", pdf_files)
            st.info(f"Selected PDF: {selected_pdf}")
            
            # Extract text from the selected PDF
            text = extract_text_from_pdf(selected_pdf)
            
            # Generating a question from the extracted text using GPT-4
            question = get_questions_from_gpt(text)
            st.write("Question: " + question)
            
            # Creating a text input for the user to ask a question
            user_question = st.text_input("Ask a question about the document")
            
            if user_question:
                # Generating an answer to the user's question using GPT-4
                answer = get_answers_from_gpt(text, user_question)
                st.write("Answer: " + answer)
                if st.button("Copy Answer Text"):
                    pyperclip.copy(answer)
                    st.success("Answer text copied to clipboard!")

# Running the main function if the script is being run directly
if __name__ == '__main__':
    main()
