import os
import streamlit as st
from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# --------------------------
# Handle API key safely
# --------------------------
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
elif "OPENAI_API_KEY" not in os.environ:
    st.error("❌ No OpenAI API key found. Please add it to Streamlit secrets.")
    st.stop()

# Initialize embeddings & LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini")  # you can use gpt-3.5-turbo if cheaper

# --------------------------
# File processing function
# --------------------------
def process_file(file_path, query):
    # Choose loader based on file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif file_path.endswith(".docx"):
        doc = DocxDocument(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        with open("temp_docx.txt", "w", encoding="utf-8") as f:
            f.write(text)
        loader = TextLoader("temp_docx.txt")
    else:
        raise ValueError("Unsupported file format")

    docs = loader.load()

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    # Vector store
    vectorstore = Chroma.from_documents(split_docs, embeddings)

    # Retrieval QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa.run(query)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="📄 PDF/Docx/Txt/CSV Q&A", layout="wide")

st.title("📚 File Q&A Chatbot")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "csv", "docx"])
user_input = st.text_input("Ask a question:")

if uploaded_file and user_input:
    # Save uploaded file
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing..."):
        try:
            result = process_file(file_path, user_input)
            st.success("✅ Answer:")
            st.write(result)
        except Exception as e:
            st.error(f"Error: {str(e)}")
