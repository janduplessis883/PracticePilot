import streamlit as st
import pdfplumber
import time
import uuid
from datetime import datetime, date
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import streamlit_shadcn_ui as ui
import weave

# New Pinecone import for the current API
from pinecone import Pinecone, ServerlessSpec

# Updated LangChain Community imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

from semantic_chunker import prepare_documents_with_semantic_chunker, clean_text, count_tokens_in_string
from pinecone_module import upload_documents_to_pinecone
from query_module import query_vector_database, generate_augmented_response
from list_docs import add_doc_googlesheet

# Initialize Weave only once
if "weave_initialized" not in st.session_state:
    weave.init('practicepilot')
    st.session_state["weave_initialized"] = True

st.set_page_config(page_title="PracticePilot")
st.logo("images/title.png", size='large')

# Set API keys securely & System variables
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
index_name = "practicepilot"
embed_model = "text-embedding-ada-002"

# 🅾️ Set Developer Mode
if "developer_mode" not in st.session_state:
    st.session_state["developer_mode"] = False

# Initialize OpenAI Client
from openai import OpenAI
if "openai_client" not in st.session_state:
    st.session_state["openai_client"] = OpenAI(api_key=openai_api_key)
client = st.session_state["openai_client"]

# Initialize GSheets connection
from streamlit_gsheets import GSheetsConnection
gsheets = GSheetsConnection(...)
conn = st.connection("gsheets", type=GSheetsConnection)

# Reload app button
clear_button = st.sidebar.button(":material/quick_phrases: Start New Chat")
if clear_button:
    st.session_state["messages"] = []
    st.cache_resource.clear()
    st.cache_data.clear()
    st.experimental_rerun()

tabs = ui.tabs(
    options=[
        "Chat with PracticePilot",
        "Upload Documents",
        "Knowledge",
        "About",
    ],
    default_value="Chat with PracticePilot",
    key="tab3",
)

# ✅ **Initialize Pinecone using the current API**
pc = Pinecone(api_key=pinecone_api_key, environment="us-east-1")

# Check if the index exists; create it if needed
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust based on your embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Wait until the index is ready
    while not pc.describe_index(index_name).status.get("ready", False):
        time.sleep(0.5)

# Get the index and set up embeddings and vector store
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model=embed_model)
vector_store = LangchainPinecone(index, embeddings, text_key="text")

# 🔥 **Chat Tab**
if tabs == "Chat with PracticePilot":
    st.image("images/chat.png")
    st.caption("Ask :material/robot_2: **PracticePilot** anything!")

    st.sidebar.header(":material/contact_support: Prompt Suggestions:")
    st.sidebar.caption(":material/prompt_suggestion: Give me an overview of the most recent GP Federation Webinar discussion points.")
    st.sidebar.caption(":material/prompt_suggestion: What were the discussion points at the most recent targets meeting?")
    st.sidebar.caption(":material/prompt_suggestion: A housebound patient requires an ECG, how can I arrange this?")

    st.sidebar.header(':material/settings: Chat Settings')
    filter_date = st.sidebar.date_input("Only consider **knowledge after**:", value=date(2024, 6, 1))
    top_k = st.sidebar.number_input("Number of vectors to return (**top_k**):", value=10, min_value=1, max_value=20)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    messages_container = st.container()
    with messages_container:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_input = st.chat_input("Ask me anything about local healthcare!")
    if user_input:
        try:
            result = query_vector_database(client, pinecone_api_key, index_name, embed_model, user_input, top_k)
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with messages_container:
                with st.chat_message("user"):
                    st.markdown(user_input)
            bot_response = generate_augmented_response(client, result, user_input)
            st.session_state["messages"].append({"role": "assistant", "content": bot_response})
            with messages_container:
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# 🔥 **Upload Documents Tab**
elif tabs == "Upload Documents":
    st.image("images/header.png")
    st.header(":material/upload: Upload Documents")

    uploaded_file = st.file_uploader("Select a **file to upload**:", type=["pdf", "txt", "md"])
    submit_button = st.button(":material/cloud_upload: Upload Document")

    if submit_button and uploaded_file:
        raw_doc = ""
        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    raw_doc += page.extract_text()
        else:
            raw_doc = uploaded_file.read().decode("utf-8")

        with st.spinner("Text pre-processing..."):
            text_doc = clean_text(raw_doc)

        with st.spinner("Semantic Chunking..."):
            documents = prepare_documents_with_semantic_chunker(text_doc, uploaded_file.name, uploaded_file.size)

        with st.spinner("Uploading to Pinecone Vector DB..."):
            for document in documents:
                doc_id = str(uuid.uuid4())
                vector_store.add_texts(
                    texts=[document["text"]],
                    metadatas=[document["metadata"]],
                    ids=[doc_id],
                )

        st.success("✅ Upload complete!")
        st.experimental_rerun()

# 🔥 **Knowledge Tab**
elif tabs == "Knowledge":
    st.image("images/header.png")
    st.header(":material/school: Knowledge")

    data = conn.read(worksheet="Sheet1", ttl="5")
    data['Publish Date'] = pd.to_datetime(data['Publish Date'])
    data.set_index('Publish Date', inplace=True)

    # Weekly aggregation for visualization
    weekly_data = data.resample("D").agg({"File Size": "sum"})
    weekly_data.reset_index(inplace=True)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(x="Publish Date", y="File Size", data=weekly_data, color="#eb6849", linewidth=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Date")
    plt.ylabel("File Size (sum)")
    plt.title("Weekly Data Upload to Vector Database")
    plt.tight_layout()
    st.pyplot(fig)

    # Filter by category
    selector = st.multiselect("Filter knowledge **by Category**:", options=list(data["Category"].unique()), default=list(data["Category"].unique()))
    search_text = st.text_input("Search Filename or Description:", "")

    if search_text:
        filtered_data = data[data["Filename"].str.contains(search_text, case=False, na=False)]
    else:
        filtered_data = data[data["Category"].isin(selector)]

    st.dataframe(filtered_data)

# 🔥 **About Tab**
elif tabs == "About":
    st.image("images/logo3.png")
    st.header(":material/privacy_tip: About")
    st.caption("The tech behind this app, and how to get this most out of your chats.")

    st.markdown("""### What is RAG (Retrieval-Augmented Generation)?
RAG stands for Retrieval-Augmented Generation, a powerful approach that combines the strengths of natural language generation and retrieval. In RAG, a pre-trained generator model is used to generate text based on input queries, while a retriever model is used to select relevant documents from a database. This hybrid approach enables the system to produce more accurate and informative results by leveraging both the contextual understanding of the generator and the precision of the retriever.
""")
    st.image("images/rag.png")
    st.markdown("""
### How do Embeddings work in Cosine Search?
In traditional search systems, the similarity between documents is often measured using metrics like TF-IDF (Term Frequency-Inverse Document Frequency). However, these methods have limitations when dealing with high-dimensional vector spaces. That's where embeddings come in. By representing each document as a dense vector, we can leverage advanced machine learning algorithms to learn meaningful patterns and relationships within the data. In cosine search, which is used by PracticePilot, embeddings play a crucial role in determining similarity between documents. The cosine similarity metric measures the angle between two vectors in a high-dimensional space, providing a more nuanced understanding of semantic relationships.
### The Power of RAG and Embeddings in PracticePilot
By integrating Retrieval-Augmented Generation with our Pinecone Vector database, we're able to provide an unparalleled search experience for healthcare professionals. Our system can generate contextualized answers, recall relevant information from large databases, and even detect biases in medical literature. The combination of the retrieval-augmented generation approach and cosine search enables us to deliver accurate, informative results that go beyond traditional search engines. This is made possible by the power of embeddings, which enable our system to capture subtle patterns and relationships within the data, ultimately enhancing the user experience for healthcare professionals.
### Chatting with PracticePilot: Tips for Success
When chatting with a RAG (Retrieval-Augmented Generation) system, keep in mind the following tips to get the most out of your conversation:
- Be specific and clear about what you're looking for
- Use simple language and avoid jargon or technical terms unless necessary
- Ask follow-up questions or request additional information if needed

To further verify the accuracy of results, Practice Pilot displays the source of the embedded text along with its corresponding cosine score. This allows you to check the credibility of the information and see where it was sourced from. By taking advantage of this feature, you can increase your confidence in the system's output.
If you still can't find an answer, try rephrasing your question to better match the system's capabilities and understanding.
""")
    st.markdown("![Static Badge](https://img.shields.io/badge/GitHub%20-%20janduplessis883%20-%20%23e3974a)")
    if st.session_state['developer_mode']:
        st.subheader("Session State")
        st.write(st.session_state)
