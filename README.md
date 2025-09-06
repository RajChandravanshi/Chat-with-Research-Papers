# 🛠️ Steps to Build a Research Paper Chatbot

This project builds a **chatbot for research papers** that lets you upload PDFs, ask questions, and get structured summaries.  
Here’s the step-by-step process:  

---

## 1. 📂 Collect Input (Upload PDFs)  
- Allow users to upload one or more research papers in PDF format.  
- Extract raw text using **PyPDF2**.  

---

## 2. 📝 Preprocess the Text  
- Clean and combine extracted text.  
- Split text into **chunks** (e.g., 1000 characters with 200 overlap).  
- Helps the model handle large documents effectively.  

📌 Tool: `LangChain CharacterTextSplitter`

---

## 3. 🔎 Create Embeddings  
- Convert text chunks into **vector embeddings** (numerical meaning).  
- Example models:  
  - `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)  
  - OpenAI embeddings  

📌 Tool: `HuggingFaceEmbeddings`

---

## 4. 📦 Store Vectors in a Database  
- Store embeddings in a **Vector Database** (e.g., FAISS).  
- Enables **semantic search** to retrieve the most relevant text chunks.  

📌 Tool: `FAISS`

---

## 5. 🤖 Choose a Language Model  
- Use a **Large Language Model (LLM)** for Q&A.  
- Examples:  
  - HuggingFace models (`moonshotai/Kimi-K2-Instruct`)  
  - OpenAI GPT models  
- Connect LLM with retriever for **context-aware answers**.  

📌 Tool: `LangChain ConversationalRetrievalChain`

---

## 6. 💬 Add Conversational Memory  
- Store **chat history** to maintain context between questions.  
- Example: `ConversationBufferMemory` in LangChain.  

---

## 7. 🎨 Build User Interface  
- Create a **Streamlit app**:  
  - Sidebar for uploading and processing PDFs  
  - Input box for user queries  
  - Chat-like display for user ↔ bot conversation  

📌 Tools: `Streamlit` + `Custom HTML Templates`

---

## 8. 📄 Add Summarization  
- Add a feature to **summarize the entire paper** into:  
  - Title  
  - Objective  
  - Methodology  
  - Key Findings  
  - Conclusion  

📌 Tool: HuggingFaceEndpoint + Prompt Engineering
![Logo](./image/Chat with Paper.png)
