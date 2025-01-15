import time
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import uuid

# Initialize Pinecone client
#api_key = st.secrets["PINECONE_API_KEY"]


# Function to upload documents to Pinecone
def upload_documents_to_pinecone(documents, vector_store):
    """
    Uploads pre-processed chunked documents to a Pinecone vector store.

    Args:
        documents (list): A list of dictionaries with "text" and "metadata".
                          Example:
                          [
                              {"text": "chunk of text 1", "metadata": {"file_name": "example.txt", "date": "2025-01-14"}},
                              {"text": "chunk of text 2", "metadata": {"file_name": "example.txt", "date": "2025-01-14"}}
                          ]
        vector_store (PineconeVectorStore): The initialized Pinecone vector store.

    Returns:
        None
    """
    for document in documents:
        # Generate a unique ID for each chunk
        doc_id = str(uuid.uuid4())

        # Extract text and metadata
        text = document["text"]
        metadata = document["metadata"]

        # Embed and upsert the document
        vector_store.add_texts(
            texts=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )



    # Upload documents to Pinecone

    st.success("✅ Documents successfully uploaded to Pinecone.")
