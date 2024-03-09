# app.py
from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('tip_predictor_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        total_bill = float(request.form['total_bill'])
        size = int(request.form['size'])
        
        # Create the feature array
        features = np.array([[total_bill, size]])
        prediction = model.predict(features)
        
        return render_template('results.html', prediction=round(prediction[0], 2))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
