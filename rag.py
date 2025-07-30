import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings #to get embedding model
from langchain.schema import Document # to store text and metadata
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS # to store the embedded data and similarity search
key=os.getenv('gapi_key')
genai.configure(api_key=key)
gemini_model=genai.GenerativeModel('gemini-2.0-flash')
def load_embedding():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
with st.spinner('Loading Embedding Model...'):
    embedding_model=load_embedding()
st.header('RAG Assistant :blue[Using Embedding & Gemini LLM]')
st.subheader('Your Intelligent Document Assistant!')
st.sidebar.text('Designed by Varshini J')
uploaded_file=st.file_uploader('Upload a PDF document here',type=['PDF'])
if uploaded_file:
    st.write('Uploaded successfully')
if uploaded_file:
    pdf=PdfReader(uploaded_file)
    raw_text=''
    for page in pdf.pages:
        raw_text+=page.extract_text()
    st.write('Extracted successfully')
    if raw_text.strip():
        doc=Document(page_content=raw_text)
        splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunk_text=splitter.split_documents([doc])
        text=[i.page_content for i in chunk_text]
        vector_db=FAISS.from_texts(text,embedding_model)
        retreive=vector_db.as_retriever()
        st.success("Doc Processed,Ask questions now")
        query=st.text_input("Enter your query here :")
        if query:
            with st.chat_message('human'):
                with st.spinner('Analyzing the document...'):
                    relv_docs=retreive.get_relevant_documents(query)
                    content='\n\n'.join([i.page_content for i in relv_docs])
                    prompt=f''' You are an AI expert.Use the content to give the answer the query asked by the user,if you are unsure tell that you dont know with a sad emoji.'
                    Content:{content}
                    Query:{query}
                    Result:'''
                    response=gemini_model.generate_content(prompt)
                    st.markdown(':green[Result]')
                    st.write(response.text)
    else:
        st.warning('Drop the file in proper format.')