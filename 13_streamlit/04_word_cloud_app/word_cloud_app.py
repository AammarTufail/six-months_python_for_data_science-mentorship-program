import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import PyPDF2
from docx import Document
import plotly.express as px
import base64
from io import BytesIO

# Functions for file reading
def read_txt(file):
    return file.getvalue().decode("utf-8")

def read_docx(file):
    doc = Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() for page in pdf.pages])

# Function to filter out stopwords
def filter_stopwords(text, additional_stopwords=[]):
    words = text.split()
    all_stopwords = STOPWORDS.union(set(additional_stopwords))
    filtered_words = [word for word in words if word.lower() not in all_stopwords]
    return " ".join(filtered_words)

# Function to create download link for plot
def get_image_download_link(buffered, format_):
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f'<a href="data:image/{format_};base64,{image_base64}" download="wordcloud.{format_}">Download Plot as {format_}</a>'

# Function to generate a download link for a DataFrame
def get_table_download_link(df, filename, file_label):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{file_label}</a>'

# Streamlit code
st.title("Word Cloud Generator")
st.subheader("📁 Upload a pdf, docx or text file to generate a word cloud")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
st.set_option('deprecation.showPyplotGlobalUse', False)

if uploaded_file:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)

    # Check the file type and read the file
    if uploaded_file.type == "text/plain":
        text = read_txt(uploaded_file)
    elif uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = read_docx(uploaded_file)
    else:
        st.error("File type not supported. Please upload a txt, pdf or docx file.")
        st.stop()

    # Generate word count table
    words = text.split()
    word_count = pd.DataFrame({'Word': words}).groupby('Word').size().reset_index(name='Count').sort_values('Count', ascending=False)

    # Sidebar: Checkbox and Multiselect box for stopwords
    use_standard_stopwords = st.sidebar.checkbox("Use standard stopwords?", True)
    top_words = word_count['Word'].head(50).tolist()
    additional_stopwords = st.sidebar.multiselect("Additional stopwords:", sorted(top_words))

    if use_standard_stopwords:
        all_stopwords = STOPWORDS.union(set(additional_stopwords))
    else:
        all_stopwords = set(additional_stopwords)

    text = filter_stopwords(text, all_stopwords)

    if text:
        # Word Cloud dimensions
        width = st.sidebar.slider("Select Word Cloud Width", 400, 2000, 1200, 50)
        height = st.sidebar.slider("Select Word Cloud Height", 200, 2000, 800, 50)

        # Generate wordcloud
        st.subheader("Generated Word Cloud")
        fig, ax = plt.subplots(figsize=(width/100, height/100))  # Convert pixels to inches for figsize
        wordcloud_img = WordCloud(width=width, height=height, background_color='white', max_words=200, contour_width=3, contour_color='steelblue').generate(text)
        ax.imshow(wordcloud_img, interpolation='bilinear')
        ax.axis('off')

        # Save plot functionality
        format_ = st.selectbox("Select file format to save the plot", ["png", "jpeg", "svg", "pdf"])
        resolution = st.slider("Select Resolution", 100, 500, 300, 50)
        # Generate word count table
        st.subheader("Word Count Table")
        words = text.split()
        word_count = pd.DataFrame({'Word': words}).groupby('Word').size().reset_index(name='Count').sort_values('Count', ascending=False)
        st.write(word_count)
    st.pyplot(fig)
    if st.button(f"Save as {format_}"):
        buffered = BytesIO()
        plt.savefig(buffered, format=format_, dpi=resolution)
        st.markdown(get_image_download_link(buffered, format_), unsafe_allow_html=True)
    
    
    # Word count table at the end
    st.sidebar.markdown("---")
    st.sidebar.subheader("Subscribe to our Youtube Channel to learn Data Science in Urdu/Hindi")
    # add a youtube video
    st.sidebar.video("https://youtu.be/omk5b1m2h38")
    st.sidebar.markdown("---")
    # add author name and info
    st.sidebar.markdown("Created by: [Dr. Muhammad Aammar Tufail](https://github.com/AammarTufail)")
    st.sidebar.markdown("Contact: [Email](mailto:aammar@codanics.com)")


    
    
    st.subheader("Word Count Table")
    st.write(word_count)
    # Provide download link for table
    if st.button('Download Word Count Table as CSV'):
        st.markdown(get_table_download_link(word_count, "word_count.csv", "Click Here to Download"), unsafe_allow_html=True)
