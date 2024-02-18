import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO

# Function to generate a download link for dataframes
def get_table_download_link(df, filename='data.csv', text='Download CSV file'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to save plot
def get_image_download_link(fig, filename='plot.png', text='Download Plot'):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Streamlit app setup
st.title('Dynamic Plot Generator with EDA')

# Sidebar options
dataset_name = st.sidebar.selectbox('Select dataset:', ['titanic', 'iris', 'tips', 'flights'])
df = sns.load_dataset(dataset_name)

if st.sidebar.checkbox('Show raw data'):
    st.write(df.head())

# Choose columns
available_columns = df.columns.tolist()
x_axis = st.sidebar.selectbox('Choose X-axis:', available_columns, index=0)
y_axis_options = ['None'] + available_columns
y_axis = st.sidebar.selectbox('Choose Y-axis (optional for some plots):', y_axis_options, index=0)
y_axis = None if y_axis == 'None' else y_axis

# Choose plot type
plot_type = st.sidebar.selectbox('Select plot type:', ['Count Plot', 'Box Plot', 'Histogram', 'Scatter Plot'])

# Generate plot based on selection
st.subheader(f'{plot_type} for {dataset_name} dataset')

fig, ax = plt.subplots()

if plot_type == 'Count Plot':
    if y_axis is None:
        sns.countplot(x=x_axis, data=df, ax=ax)
    else:
        sns.countplot(x=x_axis, hue=y_axis, data=df, ax=ax)
elif plot_type == 'Box Plot' and y_axis is not None:
    sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)
elif plot_type == 'Histogram':
    sns.histplot(df[x_axis], kde=True, ax=ax)
elif plot_type == 'Scatter Plot' and y_axis is not None:
    sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax)

st.pyplot(fig)
st.markdown(get_image_download_link(fig), unsafe_allow_html=True)

# Basic EDA
if st.button('Show Basic EDA'):
    st.write(df.describe())
    st.markdown(get_table_download_link(df.describe()), unsafe_allow_html=True)
