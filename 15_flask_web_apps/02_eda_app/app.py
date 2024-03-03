from flask import Flask, render_template
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry non-GUI backend suited for scripts and web environments
import matplotlib.pyplot as plt

import io
import base64

app = Flask(__name__)

def get_iris_data():
    # Load the Iris dataset
    iris = sns.load_dataset('iris')
    return iris

def plot_iris_data(df):
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
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    return plot_url

@app.route('/')
def index():
    # Load and plot the Iris dataset
    iris_data = get_iris_data()
    plot_url = plot_iris_data(iris_data)
    
    # Render the plot in the template
    return render_template("index.html", plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
