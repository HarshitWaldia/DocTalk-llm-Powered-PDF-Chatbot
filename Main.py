import streamlit as st
import os
import PyPDF2
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if groq_api_key is None:
    st.error("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")
    st.stop()
if google_api_key is None:
    st.error("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("Chat with PDF 💬")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions: {input}
"""
)

def vector_embedding(uploaded_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    loaded_docs = [pdf_reader.pages[page_num].extract_text() for page_num in range(len(pdf_reader.pages))]
    return loaded_docs

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    vector_store = vector_embedding(uploaded_file)
    st.write("Vector Store DB Is Ready")

import time

# Create a form for the search input and button
with st.form(key='search_form'):
    prompt1 = st.text_input("Enter Your Question From Documents")
    submit_button = st.form_submit_button(label='Search')

if submit_button:
    if uploaded_file is not None and prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = FAISS.from_texts(vector_store, GoogleGenerativeAIEmbeddings(model="models/embedding-001")).as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc)
                st.write("--------------------------------")

with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Gemma](https://github.com/Langchain/gemma)
    ''')
    st.write('Made with ❤️ by Harshit')
