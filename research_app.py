import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# ------------------------
# Load environment variables
# ------------------------
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ------------------------
# Helper Functions
# ------------------------
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Avoid NoneType errors
                text += page_text
    return text

def get_text_chunks(text):
    """Split text into chunks for embedding."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    """Create FAISS vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

def create_conversation_chain(vectorstore):
    """Create a Conversational Retrieval Chain with memory."""
    """Build conversational retrieval chain with Hugging Face LLM"""
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="moonshotai/Kimi-K2-Instruct",
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        temperature=0.7,
        max_new_tokens=256
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

def handle_user_input(user_question):
    """Handle user input and show chat history."""
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # user
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:  # bot
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def summarize_research_paper(text):
    """Summarize research paper into structured sections."""
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

    Be concise but informative. Use bullet points if helpful.

    Research paper text:
    {text}
    """
    summary = llm.predict(prompt)
    return summary

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Research Paper Q&A + Summarizer", page_icon="📚", layout="centered")
st.write(css, unsafe_allow_html=True)

st.header("📚 Research Paper Q&A + Summarizer")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

# User input for questions
user_question = st.text_input("Ask a question about your research paper:")
if user_question and st.session_state.conversation is not None:
    handle_user_input(user_question)
elif user_question:
    st.warning("⚠️ Please upload and process your research paper first.")

# Sidebar - PDF Upload
with st.sidebar:
    st.subheader("📂 Your Research Papers")
    pdf_docs = st.file_uploader(
        "Upload your research papers here and click 'Process'",
        accept_multiple_files=True,
        type=["pdf"]
    )

    if st.button("Process") and pdf_docs:
        with st.spinner("Processing research paper(s)..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = create_conversation_chain(vectorstore)
        st.success("✅ Processing complete! You can now ask questions.")

    if st.button("Summarize Research Paper") and pdf_docs:
        with st.spinner("Summarizing research paper..."):
            raw_text = get_pdf_text(pdf_docs)
            summary = summarize_research_paper(raw_text)
        st.subheader("📄 Research Paper Summary")
        st.write(summary)
