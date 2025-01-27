# PracticePilot

### RAG Solutions for Brompton Health PCN

![logo](images/logo.png)

# Overview

This project leverages the power of Retrieval-Augmented Generation (RAG) and cosine search to provide an unparalleled search experience for healthcare professionals. Our system combines the strengths of natural language generation and retrieval to deliver accurate, informative results that go beyond traditional search engines.

## Techniques Implemented

### 1. Retrieval-Augmented Generation

Our system utilizes RAG, a hybrid approach that combines the strengths of natural language generation and retrieval. We use a pre-trained generator model to generate text based on input queries, while a retriever model is used to select relevant documents from our Pinecone Vector database.

### 2. Cosine Search

We employ cosine search, which measures the similarity between documents by computing the cosine similarity metric between vectors in a high-dimensional space. This enables us to capture subtle patterns and relationships within the data, ultimately enhancing the user experience for healthcare professionals.

### 3. Embeddings

Our system utilizes embeddings, which represent each document as a dense vector. We leverage advanced machine learning algorithms to learn meaningful patterns and relationships within the data, providing a more nuanced understanding of semantic relationships.

### 4. Natural Language Processing (NLP)

We implement various NLP techniques, such as tokenization, stemming, and lemmatization, to preprocess input queries and improve search results.

# Use Cases

*   **Healthcare Professionals**: Our system provides an unparalleled search experience for healthcare professionals, enabling them to quickly find relevant information and answer complex questions.
*   **Medical Literature Review**: We can help medical literature review by providing a robust search interface that retrieves relevant documents based on user queries.

# Limitations

*   **Limited Domain Knowledge**: While our system is designed to handle various healthcare domains, it may not perform optimally for specific or niche areas.
*   **Query Complexity**: The complexity of user queries can impact search results; more complex queries may require additional preprocessing or refinement.

# Future Work

*   **Improving RAG Model Performance**: We plan to refine our RAG model by incorporating additional techniques, such as knowledge graph-based retrieval, entity recognition, **re-ranking** and **para-phrasing**. See Issues for progress.
*   **Expanding Domain Knowledge**: We aim to expand our domain knowledge by incorporating additional healthcare domains and entities.
