import time
import streamlit as st
import uuid

import weave

# Function to upload documents to Pinecone with a progress bar
@weave.op()
def upload_documents_to_pinecone(documents, vector_store):
    """
    Uploads pre-processed chunked documents to a Pinecone vector store with a progress bar.

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
    total_documents = len(documents)
    progress_bar = st.progress(0)  # Initialize progress bar
    progress_text = st.empty()  # Create an empty text element for updates

    for i, document in enumerate(documents):
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

        # Update progress bar and text
        progress = (i + 1) / total_documents
        progress_bar.progress(progress)
        progress_text.text(f"Uploading document {i + 1} of {total_documents}...")


        # Simulate a slight delay for better UI experience (remove in production)
        time.sleep(0.1)

    # Mark completion
    progress_text.text("Upload complete!")
    st.success("✅ Documents successfully uploaded to Pinecone.")
