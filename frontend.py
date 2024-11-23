import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import streamlit.components.v1 as components

# Base URL of your Flask backend
BASE_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="Data Visualization App", layout="wide")

# Helper function to handle API requests
def call_api(endpoint, method="GET", files=None, json_data=None):
    try:
        url = f"{BASE_URL}{endpoint}"
        if method == "POST":
            if files:
                response = requests.post(url, files=files)
            elif json_data:
                response = requests.post(url, json=json_data)
            else:
                response = requests.post(url)
        else:
            response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

# Function to render different chart types
def render_chart(chart_type, variables, data):
    if isinstance(data, list):
        data = pd.DataFrame(data)  # Convert to DataFrame if it's a list of dicts

    if chart_type == "bar":
        fig = px.bar(data, x=variables[0], y=variables[1], title=f"{variables[0]} vs {variables[1]}")
    elif chart_type == "scatter":
        fig = px.scatter(data, x=variables[0], y=variables[1], title=f"{variables[0]} vs {variables[1]}")
    elif chart_type == "line":
        fig = px.line(data, x=variables[0], y=variables[1], title=f"{variables[0]} vs {variables[1]}")
    elif chart_type == "pie":
        fig = px.pie(data, names=variables[0], values=variables[1], title=f"{variables[0]} Distribution")
    elif chart_type == "heatmap":
        fig = px.imshow(data[variables], title=f"Heatmap for {', '.join(variables)}")
    elif chart_type == "box":
        fig = px.box(data, y=variables[0], title=f"Boxplot for {variables[0]}")
    elif chart_type == "violin":
        fig = px.violin(data, y=variables[0], title=f"Violin plot for {variables[0]}")
    else:
        st.error(f"Unsupported chart type: {chart_type}")
        return None

    st.plotly_chart(fig)


# App sections
def upload_dataset():
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file to upload", type=["csv"])

    if uploaded_file:
        files = {"file": uploaded_file}
        response = call_api("/upload-dataset", method="POST", files=files)
        if response:
            st.success(response.get("message"))
            st.write(f"Dataset shape: {response.get('shape', '')}")


def analyze_dataset():
    st.header("Analyze Dataset")
    response = call_api("/analyze-dataset")

    if response:
        st.json(response)
    else:
        st.error("Failed to analyze the dataset. Please ensure a dataset is uploaded.")


def get_visualizations():
    st.header("Get Visualization Suggestions")
    response = call_api("/get-visualizations")

    if response:
        visualization_suggestions = response.get("visualization_suggestions", [])
        st.subheader("Visualization Suggestions")

        for suggestion in visualization_suggestions:
            st.write(f"Chart Type: {suggestion['chart_type']}")
            st.write(f"Variables: {', '.join(suggestion['variables'])}")
            st.write(f"Description: {suggestion['description']}")

            # Render the chart dynamically using Plotly (via Streamlit)
            if "chart_data" in suggestion:
                render_chart(suggestion["chart_type"], suggestion["variables"], suggestion["chart_data"])
            st.write("---")
    else:
        st.error("Failed to get visualization suggestions. Please ensure a dataset is uploaded.")


def query_visualizations():
    st.header("Query Visualizations")
    query = st.text_input("Enter a natural language query about the dataset:")

    if st.button("Get Analysis for Query"):
        if query.strip():
            response = call_api("/query-visualizations", method="POST", json_data={"query": query})

            # Debugging: show the response from the backend
            if response:
                st.json(response)  # Show the raw response for debugging

                # Just display the analysis, no chart rendering here
                if 'analysis' in response:
                    st.write(f"Analysis: {response['analysis']}")
                # else:
                #     st.warning("No analysis found for the query.")
            else:
                st.error("Failed to process the query.")
        else:
            st.error("Please enter a query before submitting.")


def explore_data():
    st.header("Explore Dataset")
    response = call_api("/get-chart-data")

    if response:
        columns = response.get("columns", [])
        data = response.get("data", [])

        if data:
            df = pd.DataFrame(data, columns=columns)
            st.dataframe(df)

            st.subheader("Create Custom Chart")
            chart_type = st.selectbox("Select Chart Type", ["scatter", "bar", "pie", "line", "heatmap", "box", "violin"])
            variables = st.multiselect("Select Variables", columns)

            if st.button("Generate Chart"):
                if chart_type and variables:
                    render_chart(chart_type, variables, df)
                else:
                    st.warning("Please select valid chart type and variables.")
        else:
            st.warning("No data available in the dataset.")
    else:
        st.error("Failed to load dataset. Please ensure a dataset is uploaded.")


# Main app
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose a page:",
    ["Upload Dataset", "Explore Data", "Analyze Dataset", "Visualization Suggestions", "Query Visualizations"]
)

if option == "Upload Dataset":
    upload_dataset()
elif option == "Explore Data":
    explore_data()
elif option == "Analyze Dataset":
    analyze_dataset()
elif option == "Visualization Suggestions":
    get_visualizations()
elif option == "Query Visualizations":
    query_visualizations()
