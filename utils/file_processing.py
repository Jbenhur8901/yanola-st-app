from langchain_voyageai import VoyageAIEmbeddings
from langchain_pinecone import PineconeVectorStore
#Pdf files reader and extractors
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, DirectoryLoader
from PyPDF2 import PdfReader
from langchain_core.documents import Document
#Vector embeddings makers
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import logging
from dotenv import load_dotenv
import os
import streamlit as st
load_dotenv()

#Texts splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

embedding = VoyageAIEmbeddings(model="voyage-large-2-instruct",api_key=st.secrets["api_keys"]["voyageai_api_key"])


def process_pdf_directory(path):
    """
    Process PDF files in the specified directory.

    Args:
        path (str): The path to the directory containing PDF files.

    Returns:
        list: A list of text chunks extracted from the PDF files in the directory.
    """
    loader = DirectoryLoader(path, glob="./*.pdf", loader_cls=PyPDFLoader)
    files = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_documents(files)
    return chunks

def load_pdf(file, chunk_size=1000, chunk_overlap=300):
    
    document = []
    content = ""
    pdf_reader = PdfReader(file)
    title = pdf_reader.metadata["/Title"]
    
    for number, page in enumerate(pdf_reader.pages) :      
        content = page.extract_text()
        document.append(Document(page_content = content, metadata= {"title": title, "page" : number}))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(document)
    return chunks        

def load_docx(path, chunk_size=1000, chunk_overlap=300):
    
    loader = Docx2txtLoader(path)
    file = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(file)
    return chunks

def add_documents(index,docs):
    """
    Add documents to the Pinecone vector index.

    Parameters:
    - index: str
        The name of the Pinecone vector index where the documents will be added.
    - docs: list
        A list of documents to be added to the index.

    Returns:
    - ids: list
        A list of unique identifiers assigned to each document added to the index.
    """
    vector = PineconeVectorStore(index_name=index,embedding=embedding, pinecone_api_key=st.secrets["api_keys"]["pinecone_api_key"])
    ids = vector.add_documents(docs)
    return ids

def create_index(index_name : str , dimension : int =1024, metric : str ="cosine", cloud : str ="aws", region : str ="us-east-1"):
    
    pc = Pinecone(api_key=st.secrets["api_keys"]["pinecone_api_key"])
    index_name = index_name  # change if desired

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region)
            )
    else:
        "Failed"
    return "Success"

def delete_index(index_name):

    pc = Pinecone(api_key=st.secrets["api_keys"]["pinecone_api_key"])
    pc.delete_index(index_name)
    return "Success"

def list_index() : 

    pc = Pinecone(api_key=st.secrets["api_keys"]["pinecone_api_key"])


