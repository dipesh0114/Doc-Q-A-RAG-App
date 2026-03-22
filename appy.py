import streamlit as st
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Streamlit UI
st.title("📄 Gemma Model Document Q&A")

# Check API key
if not groq_api_key:
    st.error("❌ GROQ API key not found. Add it in .env file")
    st.stop()

# LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"   # ✅ fixed
)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Vector Embedding Function
def vector_embedding():
    if "vectors" not in st.session_state:

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

        st.session_state.loader = PyPDFDirectoryLoader("./pdffile")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:20]
        )

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

# User Input
prompt1 = st.text_input("Enter your question from documents")

# Button to create embeddings
if st.button("📚 Create Document Embeddings"):
    vector_embedding()
    st.success("✅ Vector Store DB is Ready!")

# Question Answering
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create embeddings first!")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = st.session_state.vectors.as_retriever()

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()

        response = retrieval_chain.invoke({'input': prompt1})

        st.write("### ✅ Answer:")
        st.write(response['answer'])

        st.write(f"⏱ Response Time: {time.process_time() - start:.2f} seconds")

        # Show Similar Documents
        with st.expander("📄 Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("--------------------------------")