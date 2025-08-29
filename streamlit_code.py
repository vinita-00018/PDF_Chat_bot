import os
import streamlit as st
from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from groq import Groq  # ✅ Official Groq client

# ================== CONFIG ==================
st.session_state.groq_client = Groq(api_key="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")
MODEL = "llama3-70b-8192"

# ================== MEMORY STORE ==================
file_store = {}

# ================== HELPERS ==================
def extract_text(file_path: str):
    """Extract text from supported file types."""
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == "pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])

    elif file_extension == "txt":
        loader = TextLoader(file_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])

    elif file_extension == "csv":
        loader = CSVLoader(file_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])

    elif file_extension == "docx":
        docx_document = DocxDocument(file_path)
        return "\n".join([p.text for p in docx_document.paragraphs])

    else:
        return None


def process_file_with_groq(file_path: str, question: str):
    """Send file content + question as prompt to Groq."""
    file_name = os.path.basename(file_path)
    file_text = extract_text(file_path)

    if not file_text:
        return {file_name: "❌ Unsupported or empty file."}

    full_prompt = f"""
You are an assistant. The user has uploaded a file named {file_name}.
Here is the file content:

{file_text[:4000]}  # limit to avoid token overflow

Now answer the following question based only on the file content:
{question}
"""

    response = st.session_state.groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": full_prompt}],
    )

    return {file_name: response.choices[0].message.content}


# ================== STREAMLIT UI ==================
st.set_page_config(page_title="Groq Multi-File Chat-Bot", layout="wide")

st.title("💬 Multi-File Chat-Bot)")
st.write("Upload files (PDF, TXT, CSV, DOCX) and ask questions using Groq API directly.")

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
            result = process_file_with_groq(file_path, user_input)
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

