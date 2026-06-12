import streamlit as st

# Set the page title shown in the browser tab
st.set_page_config(page_title="PDF Question Answering System")

# Display the main heading of the application
st.title("📄 Intelligent PDF Question Answering System")

# Allow the user to upload a PDF file
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)

# Display the uploaded file name after successful upload
if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")