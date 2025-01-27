from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import re
import weave
import tiktoken


def count_tokens_in_string(input_string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Calculate the number of tokens in a string using the tiktoken library.

    Parameters:
        input_string (str): The input string to tokenize.
        encoding_name (str): The encoding name to use for tokenization (default is "text-embedding-3-small").

    Returns:
        int: The number of tokens in the input string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(input_string)

    # Return the number of tokens
    return len(tokens)





@weave.op()
def clean_text(text):
    """
    Replaces line breaks with a single space and strips leading/trailing whitespace.

    Args:
        text (str): The input string to clean.

    Returns:
        str: The cleaned string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    # Replace line breaks (\n, \r\n) with a single space and strip whitespace
    cleaned_text = ' '.join(text.splitlines()).strip()
    return cleaned_text

def remove_special_characters(text):
    """
    Removes special characters from the text, keeping only alphanumeric characters,
    spaces, and specified special characters.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    # Regex pattern to keep alphanumeric, spaces, and specified characters
    allowed_chars = r'a-zA-Z0-9\s%/\-\.\:\;\'\"\+\(\)\&\$£@~{}\[\]=\?\|'
    pattern = rf'[^{allowed_chars}]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text



@weave.op()
def prepare_documents_with_semantic_chunker(text_doc, doc_desc, file_name, file_size, category, pub_date):
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
        OpenAIEmbeddings(),
        breakpoint_threshold_type="gradient",
        breakpoint_threshold_amount=85.0,  # Adjust this value as needed
        min_chunk_size=20  # Ensure each chunk has at least 2 sentences
    )

    # Split the text into semantic chunks
    docs = text_splitter.create_documents([text_doc])


    # Attach metadata to each chunk
    documents_with_metadata = [
        {
            "text": doc.page_content,
            "metadata": {
                "date": pub_date,
                "file_name": file_name,
                "doc_desc": doc_desc,
                "category": category,
                "file_size": file_size,
            }
        }
        for doc in docs
    ]

    return documents_with_metadata
