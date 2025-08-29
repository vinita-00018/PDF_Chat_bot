import os
import streamlit as st
from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# --------------------------
# Load API Key from secrets
# --------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please add OPENAI_API_KEY in your Streamlit secrets!")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize embeddings & LLM
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="PDF/Doc/CSV Chatbot", layout="wide")
st.title("📄 Chat with your Documents")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or CSV file", type=["pdf", "docx", "csv", "txt"])
user_input = st.text_input("Ask a question from the document:")

def process_file(file_path, query):
    # Detect file type & load
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        doc = DocxDocument(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        with open("temp_doc.txt", "w", encoding="utf-8") as f:
            f.write(text)
        loader = TextLoader("temp_doc.txt")
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        st.error("Unsupported file type")
        return None

    documents = loader.load()

    # Split documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create vectorstore
    vectorstore = Chroma.from_documents(docs, embeddings)

    # RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
    )

    return qa_chain.run(query)

if uploaded_file is not None and user_input:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    result = process_file(uploaded_file.name, user_input)
    if result:
        st.write("### Answer:")
        st.write(result)
