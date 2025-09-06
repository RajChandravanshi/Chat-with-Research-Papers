import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def create_conversation_chain(vectorstore):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        temperature=0.7,
        max_new_tokens=256
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    # Display chat in GPT-style bubbles
    for i, message in enumerate(st.session_state.chat_history):
        if message.type == "human":
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)


def summarize_research_paper(text):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="moonshotai/Kimi-K2-Instruct",
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        temperature=0.7,
        max_new_tokens=256
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    prompt = f"""
    You are an expert research assistant. Summarize the following research paper
    in a structured format with these sections:
    1. Title
    2. Research Objective / Question
    3. Methodology
    4. Key Findings
    5. Conclusion

    Be concise but informative.

    Research paper text:
    {text}
    """
    return llm.predict(prompt)

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Research Paper Q&A + Summarizer", page_icon="📚", layout="wide")
st.title("📚 Research Paper Q&A + Summarizer")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar - PDF Upload
with st.sidebar:
    st.subheader("📂 Upload Research Papers")
    pdf_docs = st.file_uploader("Upload PDF(s)", accept_multiple_files=True, type=["pdf"])

    if st.button("Process") and pdf_docs:
        with st.spinner("Processing research paper(s)..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = create_conversation_chain(vectorstore)
        st.success("✅ Processing complete! You can now chat.")

# Main Panel - Chat + Summarization
if st.session_state.conversation:
    user_question = st.chat_input("Ask a question about your research paper...")
    if user_question:
        handle_user_input(user_question)

# Summarization button in main panel
if pdf_docs:
    if st.button("📄 Summarize Research Paper"):
        with st.spinner("Summarizing..."):
            raw_text = get_pdf_text(pdf_docs)
            summary = summarize_research_paper(raw_text)
        with st.expander("📄 Research Paper Summary", expanded=True):
            st.write(summary)
else:
    st.info("👆 Upload and process a research paper to start chatting.")
