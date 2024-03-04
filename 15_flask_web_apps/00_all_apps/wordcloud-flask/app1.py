from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
import PyPDF2
from PyPDF2 import PdfReader
import docx
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def generate_and_save_wordcloud(text, format):
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'wordcloud.{format}')
    plt.savefig(img_path, format=format, bbox_inches='tight', pad_inches=0)
    return img_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f'File saved to {file_path}')  # Debugging statement

        if filename.endswith('.pdf'):
            pdf_reader = PdfReader(file_path)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif filename.endswith('.docx'):
            doc = docx.Document(file_path)
            text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            text = ''

        format = request.form.get('format', 'png')
        img_path = generate_and_save_wordcloud(text, format)

        return redirect(url_for('result', filename=f'wordcloud.{format}'))

    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Print the absolute path of UPLOAD_FOLDER for debugging
    print(f'Upload folder is set to {os.path.abspath(app.config["UPLOAD_FOLDER"])}')
    app.run(debug=True)
