import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------------
# API Key Check
# --------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please add your OPENAI_API_KEY to Streamlit secrets.")
    st.stop()

# Initialize embeddings & LLM
embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])

# --------------------------
# File Processing Function
# --------------------------
def process_file(file_path, query):
    # Detect file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
        st.error("Unsupported file format. Please upload PDF, TXT, or CSV.")
        return None

    # Load & split docs
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)

    # Use FAISS instead of Chroma
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()
    results = retriever.get_relevant_documents(query)

    if not results:
        return "No relevant context found in the document."

    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm.invoke(prompt)
    return response.content

# --------------------------
# Streamlit UI
# --------------------------
st.title("📚 PDF/Text/CSV Q&A Bot")

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt", "csv"])
user_input = st.text_input("Ask a question about the document:")

if uploaded_file and user_input:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    result = process_file(uploaded_file.name, user_input)
    st.write("### Answer:")
    st.write(result)
