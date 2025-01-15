from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

def query_vector_database(client, pinecone_api_key, index_name, embed_model, query_text, top_k=5):
    """
    Queries the Pinecone vector database using OpenAI embeddings for a given query text.

    Args:
        client: OpenAI client for generating embeddings.
        pinecone_api_key (str): API key for Pinecone.
        index_name (str): Name of the Pinecone index to query.
        embed_model (str): OpenAI embedding model to use.
        query_text (str): Query text for which to generate embeddings and retrieve results.
        top_k (int): Number of top results to retrieve from the vector database.

    Returns:
        dict: Query results from the Pinecone index.
    """
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    print("Pinecone Connected")

    # Load or create the vector database
    index = pc.Index(index_name)
    print("Index Loaded.")

    # Generate embedding using OpenAI
    res = client.embeddings.create(
        input=query_text,
        model=embed_model,
        encoding_format="float",
    )
    print("OpenAI Query Embedding created")

    # Extract embedding from the response
    xq = res.data[0].embedding  # Correct way to access the embedding
    print("Query Embedding Extracted")

    # Query the Pinecone database
    result = index.query(vector=xq, top_k=top_k, include_metadata=True)
    print("Query Vector Database")
    print("Output: ")

    return result


def generate_augmented_response(client, result, query, model="gpt-4o-mini"):
    """
    Generates a response using the retrieved context from RAG retrieval and a query.

    Args:
        client (OpenAI): The OpenAI client instance.
        result (dict): The output from the RAG retrieval (Pinecone query result).
        query (str): The user's query.
        model (str): The OpenAI model to use for the chat completion. Default is "gpt-4o-mini".

    Returns:
        dict: The chat completion response.
    """
    # Extract the retrieved contexts
    contexts = [item['metadata']['text'] for item in result['matches']]

    # Combine the contexts with the query
    augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query
    print("Generated Augmented query.")
    # System message to prime the model
    primer = """You are Q&A bot. A highly intelligent system that answers
    user questions based on the information provided by the user above
    each question. If the information cannot be found in the information
    provided by the user you truthfully say '❌ I don't know'.
    """
    print("Final chat completion to OpenAI.")
    # Generate the chat completion
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query}
        ],
        model=model,
        temperature=0
    )

    return completion.choices[0].message.content
