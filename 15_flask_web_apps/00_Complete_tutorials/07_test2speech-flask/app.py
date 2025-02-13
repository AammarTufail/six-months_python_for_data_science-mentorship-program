from flask import Flask, render_template, request, send_file, url_for
import os
import tempfile
import pydub
from openai import OpenAI

app = Flask(__name__)

# Replace YOUR_API_KEY with your actual OpenAI API key
openai = OpenAI(api_key="")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    text = request.form['text']
    model = request.form['model']
    voice = request.form['voice']
    format = request.form['format']

    mp3_speech_path = text_to_speech(text, model, voice)

    if format != "mp3":
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmpfile:
            convert_audio_format(mp3_speech_path, tmpfile.name, format)
            speech_path = tmpfile.name
        os.remove(mp3_speech_path)
    else:
        speech_path = mp3_speech_path

    return send_file(speech_path, as_attachment=True)

def text_to_speech(text, model, voice):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        speech_file_path = tmpfile.name
        response = openai.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        response.stream_to_file(speech_file_path)
        return speech_file_path

def convert_audio_format(input_path, output_path, format):
    audio = pydub.AudioSegment.from_mp3(input_path)
    audio.export(output_path, format=format)

if __name__ == '__main__':
    app.run(debug=True)


# add the name option to save the fil
# make this app a multimodel app
# use responsive images
