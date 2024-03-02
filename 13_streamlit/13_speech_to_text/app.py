import streamlit as st
from openai import OpenAI
import tempfile
import os
from pytube import YouTube
from moviepy.editor import AudioFileClip
import requests

# Define the transcription functions
def transcribe_audio(client, audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription_response.text

def transcribe_uploaded_file(client, audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + audio_file.name.split('.')[-1]) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        transcription_text = transcribe_audio(client, tmp_file_path)
        st.write("Transcription:", transcription_text)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        os.remove(tmp_file_path)  # Clean up

# Sidebar for API Key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Main app title
st.title("🎙️Speech2Text with Whisper-1🤖")

# Initialize the OpenAI client with the API key from the sidebar
client = OpenAI(api_key=api_key)

# Option for users to select input type: YouTube URL, Upload Audio File, or Use Sample Audio
input_type = st.radio("Select input type", ["Upload Audio File", "Enter YouTube URL", "Use Sample Audio"])

if input_type == "Enter YouTube URL":
    youtube_url = st.text_input("Enter a YouTube URL")
    if youtube_url and api_key:
        with st.spinner("Downloading YouTube video and extracting audio..."):
            yt = YouTube(youtube_url)
            video = yt.streams.filter(only_audio=True).first()
            temp_video_path = video.download(output_path=tempfile.gettempdir())
            temp_audio_path = os.path.join(tempfile.gettempdir(), "audio.mp3")
            
            with AudioFileClip(temp_video_path) as audio_clip:
                audio_clip.write_audiofile(temp_audio_path)
            os.remove(temp_video_path)  # Clean up video file
            
            transcription_text = transcribe_audio(client, temp_audio_path)
            st.write("Transcription:", transcription_text)
            os.remove(temp_audio_path)  # Clean up audio file

elif input_type == "Upload Audio File":
    audio_file = st.file_uploader("Upload an audio file", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"])
    if audio_file is not None and api_key:
        transcribe_uploaded_file(client, audio_file)

elif input_type == "Use Sample Audio":
    sample_audio_url = "https://github.com/Rizwankaka/speech2text-Whisper-1/raw/main/audio.mp3"
    st.markdown(f"Download the sample audio file [here]({sample_audio_url}).", unsafe_allow_html=True)
    if st.button('Transcribe Sample Audio'):
        with st.spinner("Downloading and transcribing sample audio..."):
            r = requests.get(sample_audio_url)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(r.content)
                temp_audio_path = tmp_file.name
            
            transcription_text = transcribe_audio(client, temp_audio_path)
            st.write("Transcription:", transcription_text)
            os.remove(temp_audio_path)  # Clean up

# Now, place the profile information after the main interactive elements
# Adding the HTML footer
# Profile footer HTML for sidebar
sidebar_footer_html = """
<div style="text-align: left;">
    <p style="font-size: 16px;"><b>Author: 🌟 Rizwan Rizwan 🌟</b></p>
    <a href="https://github.com/Rizwankaka"><img src="https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github" alt="GitHub"/></a><br>
    <a href="https://www.linkedin.com/in/rizwan-rizwan-1351a650/"><img src="https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn"/></a><br>
    <a href="https://twitter.com/RizwanRizwan_"><img src="https://img.shields.io/badge/Twitter-Profile-blue?style=for-the-badge&logo=twitter" alt="Twitter"/></a><br>
    <a href="https://www.facebook.com/RIZWANNAZEEER"><img src="https://img.shields.io/badge/Facebook-Profile-blue?style=for-the-badge&logo=facebook" alt="Facebook"/></a><br>
    <a href="mailto:riwan.rewala@gmail.com"><img src="https://img.shields.io/badge/Gmail-Contact%20Me-red?style=for-the-badge&logo=gmail" alt="Gmail"/></a>
</div>
"""

# Render profile footer in sidebar at the "bottom"
st.sidebar.markdown(sidebar_footer_html, unsafe_allow_html=True)

# Set a background image
def set_background_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.pexels.com/photos/4097159/pexels-photo-4097159.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1);
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image()

# Set a background image for the sidebar
sidebar_background_image = '''
<style>
[data-testid="stSidebar"] {
    background-image: url("https://www.pexels.com/photo/abstract-background-with-green-smear-of-paint-6423446/");
    background-size: cover;
}
</style>
'''

st.sidebar.markdown(sidebar_background_image, unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Custom CSS to inject into the Streamlit app
footer_css = """
<style>
.footer {
    position: fixed;
    right: 0;
    bottom: 0;
    width: auto;
    background-color: transparent;
    color: black;
    text-align: right;
    padding-right: 10px;
}
</style>
"""

# HTML for the footer - replace your credit information here
footer_html = f"""
<div class="footer" style="background-color: #3498db; color: #fff; padding: 10px; border-radius: 5px;">
    <p>Credit: Dr. Aammar Tufail | Phd | Data Scientist | Bioinformatician (<a href="https://www.youtube.com/@Codanics" target="_blank" style="color: #ffffff; text-decoration: underline;">Codanics</a>)</p>
    <a href="https://github.com/AammarTufail"><img src="https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github" alt="GitHub"/></a>
    <a href="https://www.kaggle.com/muhammadaammartufail"><img src="https://img.shields.io/badge/Kaggle-Profile-blue?style=for-the-badge&logo=kaggle" alt="Kaggle"/></a>
    <a href="https://www.linkedin.com/in/dr-muhammad-aammar-tufail-02471213b/"><img src="https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn"/></a>
    <a href="https://www.youtube.com/@codanics"><img src="https://img.shields.io/badge/YouTube-Profile-red?style=for-the-badge&logo=youtube" alt="YouTube"/></a>
    <a href="https://www.facebook.com/aammar.tufail"><img src="https://img.shields.io/badge/Facebook-Profile-blue?style=for-the-badge&logo=facebook" alt="Facebook"/></a>
    <a href="https://twitter.com/aammar_tufail"><img src="https://img.shields.io/badge/Twitter-Profile-blue?style=for-the-badge&logo=twitter" alt="Twitter/X"/></a>
    <a href="https://www.instagram.com/aammartufail/"><img src="https://img.shields.io/badge/Instagram-Profile-blue?style=for-the-badge&logo=instagram" alt="Instagram"/></a>
    <a href="mailto:aammar@codanics.com"><img src="https://img.shields.io/badge/Email-Contact%20Me-red?style=for-the-badge&logo=email" alt="Email"/></a>
</div>
"""

# Combine CSS and HTML for the footer
st.markdown(footer_css, unsafe_allow_html=True)
st.markdown(footer_html, unsafe_allow_html=True)
