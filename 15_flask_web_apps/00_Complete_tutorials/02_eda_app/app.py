# import libraries
from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Create an instance of the Flask class
app = Flask(__name__)

# define a function to load the dataset
def load_data():
    df = sns.load_dataset('iris')
    return df

# define the function to plot the data
def plot(df):
    # Create a figure for the plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot distribution of each feature
    sns.histplot(data=df, x='sepal_length', kde=True, ax=axs[0, 0], color='skyblue')
    sns.histplot(data=df, x='sepal_width', kde=True, ax=axs[0, 1], color='olive')
    sns.histplot(data=df, x='petal_length', kde=True, ax=axs[1, 0], color='gold')
    sns.histplot(data=df, x='petal_width', kde=True, ax=axs[1, 1], color='teal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the plot to base64 string for HTML embedding
    b64_image = base64.b64encode(buf.read()).decode('utf-8')
    return b64_image

# define the route
@app.route('/')
def index():
    # load the dataset
    df = load_data()
    # plot the data
    b64_image = plot(df)
    return render_template('index.html', b64_image=b64_image)

# run the app
if __name__ == '__main__':
    app.run(debug=True)
