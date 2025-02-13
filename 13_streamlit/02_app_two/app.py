import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# add title
st.title('Data Analysis Application')
st.subheader('This is a simple data analysis application created by @codanics')

# create a dropdown list to choose a dataset
dataset_options = ['iris', 'titanic', 'tips', 'diamonds']
selected_dataset = st.selectbox('Select a dataset', dataset_options)

# load the selected dataset
if selected_dataset == 'iris':
    df = sns.load_dataset('iris')
elif selected_dataset == 'titanic':
    df = sns.load_dataset('titanic')
elif selected_dataset == 'tips':
    df = sns.load_dataset('tips')
elif selected_dataset == 'diamonds':
    df = sns.load_dataset('diamonds')

# button to upload custom dataset
uploaded_file = st.file_uploader('Upload a custom dataset', type=['csv', 'xlsx'])

if uploaded_file is not None:
    # process the uploaded file
    df = pd.read_csv(uploaded_file)  # assuming the uploaded file is in CSV format

# display the dataset
st.write(df)

# display the number of Rows and Column from the selected data
st.write('Number of Rows:', df.shape[0])
st.write('Number of Columns:', df.shape[1])

# display the column names of selected data with their data types
st.write('Column Names and Data Types:', df.dtypes)

# print the null values if those are > 0
if df.isnull().sum().sum() > 0:
    st.write('Null Values:', df.isnull().sum().sort_values(ascending=False))
else:
    st.write('No Null Values')

# display the summary statistics of the selected data
st.write('Summary Statistics:', df.describe())

# Create a pairplot
st.subheader('Pairplot')
# select the column to be used as hue in pairplot
hue_column = st.selectbox('Select a column to be used as hue', df.columns)
st.pyplot(sns.pairplot(df, hue=hue_column))

# Create a heatmap
st.subheader('Heatmap')
# select the columns which are numeric and then create a corr_matrix
numeric_columns = df.select_dtypes(include=np.number).columns
corr_matrix = df[numeric_columns].corr()
numeric_columns = df.select_dtypes(include=np.number).columns
corr_matrix = df[numeric_columns].corr()

from plotly import graph_objects as go

# Convert the seaborn heatmap plot to a Plotly figure
heatmap_fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                       x=corr_matrix.columns,
                                       y=corr_matrix.columns,
                                       colorscale='Viridis'))
st.plotly_chart(heatmap_fig)


