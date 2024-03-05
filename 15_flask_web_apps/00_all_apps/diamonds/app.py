from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump

# Load the diamonds dataset from seaborn
diamonds = sns.load_dataset('diamonds')

# Drop any rows with missing values
diamonds.dropna(inplace=True)

# Convert categorical variables to numerical
diamonds = pd.get_dummies(diamonds)

# Split the dataset into input and output variables
X = diamonds.drop('price', axis=1)
y = diamonds['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor model on the training set
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse:.2f}')

# Save the trained model to a file
dump(model, 'model.joblib')

# Define a function to preprocess the input data
def preprocess_data(data):
    # Preprocess the input data as necessary (e.g., convert categorical variables to numerical, normalize numerical variables, etc.)
    data = pd.get_dummies(data)
    return data.values

# Create a new Flask app
app = Flask(__name__)

# Define a Flask route to handle incoming requests and make predictions using the machine learning model
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Preprocess the input data
    data = preprocess_data(pd.DataFrame(data))

    # Make a prediction using the model
    prediction = model.predict(data)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
