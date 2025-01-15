from datetime import datetime
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


def prepare_documents_with_semantic_chunker(text_doc, doc_desc, file_name, file_size):
    """
    Splits a document into semantically meaningful chunks using SemanticChunker
    and prepares them as Document objects with metadata for Pinecone.

    Args:
        text_doc (str): The text content of the document.
        file_name (str): The name of the file.
        file_size (int): The size of the file in bytes.

    Returns:
        list: A list of Document objects with metadata.
    """
    # Initialize the SemanticChunker with OpenAIEmbeddings
    text_splitter = SemanticChunker(
        OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
    )

    # Split the text into semantic chunks
    docs = text_splitter.create_documents([text_doc])

    # Get the current date
    today_date = datetime.now().strftime('%Y-%m-%d')

    # Attach metadata to each chunk
    documents_with_metadata = [
        {
            "text": doc.page_content,
            "metadata": {
                "date": today_date,
                "file_name": file_name,
                "doc_desc": doc_desc,
                "file_size": file_size,
            }
        }
        for doc in docs
    ]

    return documents_with_metadata
