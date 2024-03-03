from flask import Flask, request, render_template, send_file
from openai import OpenAI
import openai
import requests
from io import BytesIO
import os

# Set your OpenAI API Key here
app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = "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"
client = openai.OpenAI()
def generate_image(prompt):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response
    except Exception as e:
        print(f"Error generating image: {e}")
        return None


def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    return None

@app.route('/', methods=['GET', 'POST'])
def main():
    image_url = None  # Initialize with no image URL
    if request.method == 'POST':
        prompt = request.form['prompt']
        if prompt:
            response = generate_image(prompt)
            if response and 'data' in response and len(response['data']) > 0:
                image_url = response['data'][0]['url']
            else:
                return 'Failed to generate image. Please try again.', 400
    # Pass the image URL (or None if not available) to the template
    return render_template('index.html', image_url=image_url)

@app.route('/download-image')
def download():
    image_url = request.args.get('image_url')
    if image_url:
        image_buffer = download_image(image_url)
        if image_buffer:
            return send_file(image_buffer, attachment_filename="generated_image.png", as_attachment=True, mimetype='image/png')
    return "Failed to download image.", 400

if __name__ == '__main__':
    app.run(debug=True)
