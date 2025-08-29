import os
import streamlit as st
from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# ================== CONFIG ==================
os.environ["OPENAI_API_KEY"] = "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"
MODEL = "gpt-4o-mini"

# ================== MEMORY STORE ==================
file_store = {}

# ================== HELPERS ==================
def process_file(file_path: str, question: str):
    """
    Load a single file, build vectorstore, and answer the question.
    """
    file_extension = file_path.split('.')[-1].lower()
    file_name = os.path.basename(file_path)

    if file_extension == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == "txt":
        loader = TextLoader(file_path)
    elif file_extension == "csv":
        loader = CSVLoader(file_path)
    elif file_extension == "docx":
        docx_document = DocxDocument(file_path)
        text = "\n".join([p.text for p in docx_document.paragraphs])
        new_docx_text_file = file_name + ".txt"
        with open(new_docx_text_file, "w", encoding="utf-8") as f:
            f.write(text)
        loader = TextLoader(new_docx_text_file)
    else:
        return {"error": f"Unsupported file type: {file_extension}"}

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        docs,
        embedding=OpenAIEmbeddings(),
        persist_directory="./data"
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model=MODEL, temperature=0),
        retriever=retriever,
        chain_type="stuff"
    )
    response = qa.run(question)
    return {file_name: response}


# ================== STREAMLIT UI ==================
st.set_page_config(page_title="Multi-File Chat-Bot", layout="wide")

st.title("💬 Multi-File Chat-Bot")
st.write("Upload files (PDF, TXT, CSV, DOCX) and ask questions in a chat-like format.")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Upload Section ---
uploaded_files = st.file_uploader("Upload your files", type=["pdf", "txt", "csv", "docx"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("uploads", exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join("uploads", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        file_store[file.name] = file_path
    st.success(f"Uploaded {len(uploaded_files)} files successfully!")

# --- Chat Section ---
if file_store:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question:")
        submit = st.form_submit_button("Send")

    if submit and user_input.strip():
        responses = {}
        for file_id, file_path in file_store.items():
            result = process_file(file_path, user_input)
            responses.update(result)

        # Save to chat history
        st.session_state.chat_history.append({
            "question": user_input,
            "answers": responses
        })

# --- Display Chat History ---
if st.session_state.chat_history:
    st.write("### Chat History")
    for i, chat in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}: {chat['question']}**")
        for fname, ans in chat["answers"].items():
            st.markdown(f"- 📘 *{fname}*: {ans}")

# --- Clear Chat Button ---
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")
