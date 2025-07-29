import boto3
import streamlit as st
import os
import uuid
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import faiss

## s3_client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

bedrock_client = boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

####################################################################################

### unique id
def get_unique_id():
    return str(uuid.uuid4())

####################################################################################

### Split text into smaller chunks function
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(pages)
    return docs


####################################################################################

### create vectorstore
def create_vector_store(documents, request_id):
    vectorstore_faiss=faiss.FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True


####################################################################################


def main():
    uploaded_file = st.file_uploader("Choose a file to upload", type=["pdf"])
    if uploaded_file is not None:
        request_id  = get_unique_id()
        st.write(f"Request ID: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, "wb") as w:
            w.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()
        st.write(f"Total Pages: {len(pages)}")
        
        split_docs = split_text(pages,1000,200)
        st.write(f"Total splitted documents: {len(split_docs)}")
        
        result = create_vector_store(split_docs, request_id)
        
        st.write("Creating vector store...")
        
        if result:
            st.write("Hurray!! PDF processed successfully")
        else:
            st.write("Error!! Please check logs.")


if __name__ == "__main__":
    main()