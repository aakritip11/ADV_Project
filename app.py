import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import streamlit as st

# Download NLTK data if running for the first time
nltk.download('punkt')
nltk.download('stopwords')

# Global variable for dataset
data = None

# Preprocess user query: tokenize and remove stopwords
def preprocess_query(query):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(query.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

# Match user query tokens with dataset column names
def map_columns_to_query(tokens, dataset_columns):
    matched_columns = []
    for token in tokens:
        # Allow partial matching of the token to column names
        for column in dataset_columns:
            if token in column.lower():  # Match keywords to column names (case-insensitive)
                matched_columns.append(column)
    return matched_columns

# Identify key columns based on data type and uniqueness
def identify_key_columns(df):
    key_columns = []
    
    for column in df.columns:
        # Skip columns that have a single unique value (constant columns)
        if df[column].nunique() <= 1:
            continue
        
        # Skip columns that are likely IDs or Names
        if "id" in column.lower() or "name" in column.lower():
            continue

        # Numeric columns (int64, float64)
        if df[column].dtype in ['int64', 'float64']:
            key_columns.append((column, "numeric"))
        
        # Categorical columns (object, category)
        elif df[column].dtype == 'object' and df[column].nunique() > 1:
            key_columns.append((column, "categorical"))
    
    return key_columns

# Generate appropriate plot for key columns based on their type
def generate_plot_for_column(column, column_type):
    global data
    fig, ax = plt.subplots(figsize=(8, 6))

    if column_type == "numeric":
        # Choose one plot for numeric columns
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    elif column_type == "categorical":
        # Choose one plot for categorical columns
        sns.countplot(x=data[column], ax=ax)
        ax.set_title(f"Bar Chart of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        st.pyplot(fig)

# Answer numerical queries
def answer_numeric_query(tokens, relevant_columns):
    global data
    if len(relevant_columns) == 1:
        column = relevant_columns[0]
        if "sum" in tokens:
            return f"The sum of {column} is {data[column].sum()}."
        elif "average" in tokens or "mean" in tokens:
            return f"The average of {column} is {data[column].mean()}."
        elif "max" in tokens:
            return f"The maximum value of {column} is {data[column].max()}."
        elif "min" in tokens:
            return f"The minimum value of {column} is {data[column].min()}."
    return None

# Generate a heatmap of correlations between numeric columns
def generate_heatmap():
    global data
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 1:
        corr_matrix = data[numeric_columns].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns to generate a heatmap.")

# Streamlit app layout
def main():
    global data

    st.title("Natural Language Query Visualizer")
    st.markdown("""Upload a dataset and enter a natural language query to visualize data insights or get numeric answers.""")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
            st.dataframe(data.head())

            # Identify key columns based on uniqueness and data types (exclude irrelevant columns like name and IDs)
            key_columns = identify_key_columns(data)
            st.subheader("Key Columns and Their Data Types:")
            if not key_columns:
                st.write("No key columns found for analysis.")
            else:
                for column, col_type in key_columns:
                    st.write(f"- {column}: {col_type}")

                st.subheader("Key Columns Visualizations:")
                # Generate plots for key columns based on their data type
                for column, col_type in key_columns:
                    generate_plot_for_column(column, col_type)

            # Generate a heatmap for numeric columns correlation
            generate_heatmap()

        except Exception as e:
            st.error(f"Failed to load dataset: {e}")

    # Query input
    query = st.text_input("Enter your query:")
    if st.button("Submit Query"):
        if data is not None:
            tokens = preprocess_query(query)
            relevant_columns = map_columns_to_query(tokens, data.columns)

            if not relevant_columns:
                st.error("Could not identify relevant columns. Please refine your query.")
            else:
                # Answer numeric queries
                answer = answer_numeric_query(tokens, relevant_columns)
                if answer:
                    st.success(answer)
                else:
                    # Handle non-numeric queries for plot generation
                    st.write(f"Generating plot for column: {relevant_columns[0]}")
                    column = relevant_columns[0]
                    column_type = "numeric" if data[column].dtype in ['int64', 'float64'] else "categorical"
                    generate_plot_for_column(column, column_type)
        else:
            st.error("Please upload a dataset first.")

if __name__ == "__main__":
    main()
