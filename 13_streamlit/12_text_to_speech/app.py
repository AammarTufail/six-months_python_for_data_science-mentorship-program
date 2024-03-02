<<<<<<< HEAD
import streamlit as st
from openai import OpenAI
import tempfile
import os


# Function to convert text to speech, modified to explicitly use an API key
def text_to_speech(api_key, text: str):
    """
    Converts text to speech using OpenAI's tts-1 model and saves the output as an MP3 file,
    explicitly using an API key for authentication.
    """
    # Initialize the OpenAI client with the provided API key
    client = OpenAI(api_key=api_key)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        speech_file_path = tmpfile.name
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        # Stream the audio response to file
        response.stream_to_file(speech_file_path)
        
        # Return the path to the audio file
        return speech_file_path

# Streamlit UI setup
st.title("🔊 Text to Speech Converter 📝")
st.image("https://www.piecex.com/product_image/20190625044028-00000544-image2.png")
st.markdown("""
This app converts text to speech using OpenAI's tts-1 model. 
Please enter your OpenAI API key below. **Do not share your API key with others.**
""")

# Input for OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password")

# create a select box for a model
model = st.selectbox("Select a model", ["tts-1", "tts-1-hd"])
# create a select box for the vocal from these alloy, echo, fable, onyx, nova, and shimmer
voice = st.selectbox("Select a voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
# Text input from user

# # audio format select box from mp3 "opus", "aac", "flac", and "pcm"
# audio_format = st.selectbox("Select an audio format", ["mp3", "opus", "aac", "flac", "pcm"])


user_input = st.text_area("Enter text to convert to speech", "Hello, welcome to our text to speech converter!")

if st.button("Convert"):
    if not api_key:
        st.error("API key is required to convert text to speech.")
    else:
        try:
            speech_path = text_to_speech(api_key, user_input)
            
            # Display a link to download the MP3 file
            st.audio(open(speech_path, 'rb'), format="audio/mp3")
            
            # Clean up: delete the temporary file after use
            os.remove(speech_path)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# download button b add karen

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
            background-image: url("https://images.pexels.com/photos/6847584/pexels-photo-6847584.jpeg");
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
    background-image: url("https://images.pexels.com/photos/6101958/pexels-photo-6101958.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
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
<div class="footer">
    <p>Credit: Dr. Aammar Tufail | Phd | Data Scientist | Bioinformatician (<a href="https://www.youtube.com/@Codanics" target="_blank">CODANICS</a>)</p>
</div>
"""

# Combine CSS and HTML for the footer
st.markdown(footer_css, unsafe_allow_html=True)
st.markdown(footer_html, unsafe_allow_html=True)
=======
import streamlit as st
from openai import OpenAI
import tempfile
import os
import pydub

# Function to convert text to speech, modified to explicitly use an API key
def text_to_speech(api_key, text: str, model, voice):
    # Initialize the OpenAI client with the provided API key
    client = OpenAI(api_key=api_key)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        speech_file_path = tmpfile.name
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        # Stream the audio response to file
        response.stream_to_file(speech_file_path)

        # Return the path to the audio file
        return speech_file_path

def convert_audio_format(input_path, output_path, format):
    audio = pydub.AudioSegment.from_mp3(input_path)
    audio.export(output_path, format=format)

# Streamlit UI setup
st.title("🔊 Text to Speech Converter 📝")
st.image("https://www.piecex.com/product_image/20190625044028-00000544-image2.png")
st.markdown("""
This app converts text to speech using OpenAI's tts-1 or tts-1-hd model.
Please enter your OpenAI API key on sidebar. **Do not share your API key with others.**
""")

# Input for OpenAI API key
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# Select box for model selection
model = st.sidebar.selectbox("Select Model", ["tts-1", "tts-1-hd"])

# Select box for voice selection
voice = st.sidebar.selectbox("Select Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])

# Select box for format selection
format = st.sidebar.selectbox("Select Format", ["mp3", "opus", "aac", "flac", "wav"])

# Text input from user
user_input = st.text_area("Enter text to convert to speech", "Hello, welcome to our text to speech converter!")

if st.button("Convert"):
    if not api_key:
        st.error("API key is required to convert text to speech.")
    else:
        with st.spinner("Converting text to speech..."):
            try:
                mp3_speech_path = text_to_speech(api_key, user_input, model, voice)

                if format != "mp3":
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmpfile:
                        convert_audio_format(mp3_speech_path, tmpfile.name, format)
                        speech_path = tmpfile.name
                    os.remove(mp3_speech_path)
                else:
                    speech_path = mp3_speech_path

                # Display a link to download the audio file
                st.audio(open(speech_path, 'rb'), format=format)
                st.markdown(f'[Download {format.upper()} file]({speech_path})', unsafe_allow_html=True)

                # Clean up: delete the temporary file after use
                os.remove(speech_path)
            except Exception as e:
                st.error(f"An error occurred: {e}")


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
<div class="footer">
    <p>Credit: Dr. Aammar Tufail | Phd | Data Scientist | Bioinformatician (<a href="https://www.youtube.com/@Codanics" target="_blank">CODANICS</a>)</p>
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
>>>>>>> c8520a7f743cb39839be8556bede011f35d8fa53
