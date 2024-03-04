from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def profile():
    profile_details = {
        'name': 'Your Name',
        'bio': 'A short bio about yourself.',
        'social_media': [
            {'name': 'LinkedIn', 'url': 'https://www.linkedin.com/in/yourprofile'},
            {'name': 'GitHub', 'url': 'https://github.com/yourusername'},
            {'name': 'Twitter', 'url': 'https://twitter.com/yourusername'}
        ]
    }
    return render_template('index.html', profile=profile_details)

if __name__ == '__main__':
    app.run(debug=True)
