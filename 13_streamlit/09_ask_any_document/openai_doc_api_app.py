# Importing necessary libraries
import streamlit as st
import PyPDF2
import io
import openai
import docx2txt
import pyperclip

# Setting up OpenAI API key
openai.api_key =st.sidebar.text_input('OpenAI API Key', type='password')
#or
# openai.api_key ='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'

# Defining a function to extract text from a PDF file
def extract_text_from_pdf(file):
    # Creating a BytesIO object from the uploaded file
    pdf_file_obj = io.BytesIO(file.read())
    # Creating a PDF reader object from the BytesIO object
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    # Initializing an empty string to store the extracted text
    text = ''
    # Looping through each page of the PDF file and extracting the text
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    # Returning the extracted text
    return text

# Defining a function to extract text from a Word file
def extract_text_from_docx(file):
    # Creating a BytesIO object from the uploaded file
    docx_file_obj = io.BytesIO(file.read())
    # Extracting text from the Word file
    text = docx2txt.process(docx_file_obj)
    # Returning the extracted text
    return text

# Defining a function to extract text from a Text file
def extract_text_from_txt(file):
    # Reading the uploaded file as text
    text = file.read().decode('utf-8')
    # Returning the extracted text
    return text

# Defining a function to extract text from a file based on its type
def extract_text_from_file(file):
    # Checking the type of the uploaded file
    if file.type == 'application/pdf':
        # Extracting text from the PDF file
        text = extract_text_from_pdf(file)
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        # Extracting text from the Word file
        text = extract_text_from_docx(file)
    elif file.type == 'text/plain':
        # Extracting text from the Text file
        text = extract_text_from_txt(file)
    else:
        # Displaying an error message if the file type is not supported
        st.error("Unsupported file type!")
        text = None
    # Returning the extracted text
    return text

# Defining a function to generate questions from text using OpenAI's GPT-3
def get_questions_from_gpt(text):
    # Selecting the first 4096 characters of the text as the prompt for the GPT-3 API
    prompt = text[:4096]
    # Generating a question using the GPT-3 API
    response = openai.Completion.create(engine="gpt-3.5-turbo-instruct", prompt=prompt, temperature=0.5, max_tokens=30)
    # Returning the generated question
    return response.choices[0].text.strip()

# Defining a function to generate answers to a question using OpenAI's GPT-3
def get_answers_from_gpt(text, question):
    # Selecting the first 4096 characters of the text as the prompt for the GPT-3 API, along with the user's question
    prompt = text[:4096] + "\nQuestion: " + question + "\nAnswer:"
    # Generating an answer using the GPT-3 API
    response = openai.Completion.create(engine="gpt-3.5-turbo-instruct", prompt=prompt, temperature=0.6, max_tokens=2000)
    # Returning the generated answer
    return response.choices[0].text.strip()

# Defining the main function of the Streamlit app
def main():
    # Setting the title of the app
    st.title("Ask Questions From Uploaded Document")
    # Creating a file uploader for PDF, Word, and Text files
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
    # Checking if a file has been uploaded
    if uploaded_file is not None:
        # Extracting text from the uploaded file
        text = extract_text_from_file(uploaded_file)
        # Checking if text was extracted successfully
        if text is not None:
            # Generating a question from the extracted text using GPT-3
            question = get_questions_from_gpt(text)
            # Displaying the generated question
            st.write("Question: " + question)
            # Creating a text input for the user to ask a question
            user_question = st.text_input("Ask a question about the document")
            # Checking if the user has asked a question
            if user_question:
                # Generating an answer to the user's question using GPT-3
                answer = get_answers_from_gpt(text, user_question)
                # Displaying the generated answer
                st.write("Answer: " + answer)
                # Creating a button to copy the answer text to clipboard
                if st.button("Copy Answer Text"):
                    pyperclip.copy(answer)
                    st.success("Answer text copied to clipboard!")

# Running the main function if the script is being run directly
if __name__ == '__main__':
    main()