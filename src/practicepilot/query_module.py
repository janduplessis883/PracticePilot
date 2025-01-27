from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from datetime import datetime

# Tracking with wandb weave
import weave
# add weave decorator to track functions @weave.op()
# Get today's date
today = datetime.today()
formatted_date = today.strftime('%Y-%m-%d')

@weave.op()
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

@weave.op()
def generate_augmented_response(client, result, query, model="gpt-4o-mini"):
    """
    Generates a response using the retrieved context from RAG retrieval and a query.

    Args:
        client (OpenAI): The OpenAI client instance.
        result (dict): The output from the RAG retrieval (Pinecone query result).
        query (str): The user's query.
        model (str): The OpenAI model to use for the chat completion. Default is "gpt-4o-mini".

    Returns:
        str: The chat completion response with references to the source metadata.
    """
    # Extract the retrieved contexts with metadata
    contexts = []
    for item in result['matches']:
        text = item['metadata'].get('text', 'N/A')
        file_name = item['metadata'].get('file_name', 'Unknown File')
        publish_date = item['metadata'].get('date', 'Unknown Date')
        score = item.get('score', 'Unknown Score')

        # Combine text with metadata for reference
        context_with_metadata = (
            f"Source: {file_name}\nPublished: {publish_date}\nScore: {score}\n\n{text}"
        )
        contexts.append(context_with_metadata)

    # Combine the contexts with the query
    augmented_query = "\n\n---------\n\n".join(contexts) + "\n\n-----\n\n" + query


    with st.expander(":material/polyline: Review **Knowledge Source**/Embeddings"):
        st.code(augmented_query, wrap_lines=True)
        st.toast(f":material/share_reviews: Review **Knowledge Source** click the expander below the chat input field. `{file_name}`")



    # System message to prime the model
    primer = f"""### Task:
Respond to the user query using the provided context. For your reference today's date is {formatted_date}. Use this date for reference when you are asked questions referencing date.

### Guidelines:
- If you don't know the answer, clearly state: '😗 I don't know! Could you re-phrase your question?'
- If uncertain, or open-ended question is asked, ask the user for clarification.
- Respond in the same language as the user's query.
- If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
- If the answer isn't present in the context but you possess the knowledge, explain this to the user and provide the answer using your own understanding.
- Do not use XML tags in your response.
- Ensure citations are concise and directly related to the information provided.

### Example of Citation:
If the user asks about a specific topic and the information is found in "whitepaper.pdf", the response should include the citation at the end of your response, like so:
* "\n📄 `<file_name> - <publish_date>`."

### Output:
Provide a clear and direct response to the user's query, from the context provided."""
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
