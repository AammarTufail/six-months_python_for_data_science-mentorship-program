from flask import Flask, render_template, request, jsonify
import openai
import os

app = Flask(__name__)
openai.api_key = os.environ.get("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")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_response', methods=['POST'])
def generate_response():
    data = request.get_json()
    text = data.get('text')

    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text}
        ]
    )

    ai_text = response.choices[0].message.content
    return jsonify({'ai_text': ai_text})

if __name__ == '__main__':
    app.run(debug=True)
