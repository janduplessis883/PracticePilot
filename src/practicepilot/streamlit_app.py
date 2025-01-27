import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pdfplumber
import time
import uuid
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from datetime import datetime, date
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time


from semantic_chunker import prepare_documents_with_semantic_chunker, clean_text, count_tokens_in_string
from pinecone_module import upload_documents_to_pinecone
from query_module import query_vector_database, generate_augmented_response
from list_docs import add_doc_googlesheet
import weave

# Check if Weave is already initialized
if "weave_initialized" not in st.session_state:
    weave.init('practicepilot')  # Initialize Weave only once
    st.session_state["weave_initialized"] = True

st.set_page_config(page_title="PracticePilot")
st.logo("images/title.png", size='large')

# Set API keys securely & and System variables
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
index_name = "practicepilot"
embed_model = "text-embedding-ada-002"

# 🅾️ Set Developer Mode
if "developer_mode" not in st.session_state:
    st.session_state["developer_mode"] = True


# Check if the OpenAI client is already initialized
if "openai_client" not in st.session_state:
    st.session_state["openai_client"] = OpenAI(api_key=openai_api_key)  # Initialize OpenAI client

# Use the client from session state
client = st.session_state["openai_client"]

gsheets = GSheetsConnection(...)
conn = st.connection("gsheets", type=GSheetsConnection)

# Reload app button
clear_button = st.sidebar.button(":material/quick_phrases: Start New Chat")
if clear_button:
    st.session_state["messages"] = []
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()



# Tab layout
tabs = st.tabs([":material/robot_2: **Chat with PracticePilot**", ":material/upload: Upload Documents", ":material/school: Knowledge", ":material/privacy_tip: About"])

# Tab: Chat
with tabs[0]:
    st.subheader(":primary[:material/robot_2: Chat with PracticePilot]")
    st.caption("Ask :material/robot_2: **PracticePilot** anything! I’ve got a stash of local medical know-how ready to share. But hey, if you stump me, I’ll just have to admit it with a cheeky, ‘😕 I dunno, mate!’")

    st.sidebar.header(":material/contact_support: Prompt Suggestions:")
    st.sidebar.caption(":material/prompt_suggestion: Give me an overview of the most recent GP Federation Webinar discussion points.")
    st.sidebar.caption(":material/prompt_suggestion: What were the discussion points at the most recent targets meeting?")
    st.sidebar.caption(":material/prompt_suggestion: A housebound patient requires an ECG, how can I arrange this?")

    st.sidebar.header(':material/settings: Chat Settings')
    filter_date = st.sidebar.date_input("Only consider **knowledge after**:", value=date(2024, 6, 1), format="YYYY-MM-DD")
    top_k = st.sidebar.number_input("Number of vectors to return (**top_k**):", value=10, min_value=1, max_value=20, help="Specify how many vectors are returned.")
    st.sidebar.divider()
    st.session_state["developer_mode"] = st.sidebar.toggle("**Developer**Mode :material/code_blocks:", value=st.session_state["developer_mode"])
    st.sidebar.write(f":primary[Vector Database: :material/database: **{index_name}**]")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    messages_container = st.container(height=500, border=False)
    with messages_container:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    st.markdown("<div style='position: fixed; bottom: 0; width: 100%;'>", unsafe_allow_html=True)
    user_input = st.chat_input("Ask me anything about local healthcare!")
    st.markdown("</div>", unsafe_allow_html=True)

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

# Tab: Upload Documents
with tabs[1]:
    # Main interface
    st.image("images/header.png")
    st.header(":material/upload: Upload Documents", help="Upload new documents to the Vector Database to extend the app's knowledge.")
    st.caption("**Upload your medical documents**, such as guidelines, papers, or research, in **PDF**, **Markdown**, or **plain Text** formats. Your contributions will be chunked and processed into our Pinecone Vector database, enhancing **PracticePilot's AI-powered knowledge** and improving its ability to provide accurate answers and insights for healthcare professionals. ")

    pc = Pinecone(api_key=pinecone_api_key)

    # Check if the index exists
    if index_name not in [index["name"] for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(0.5)

    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model=embed_model)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Default values for form fields
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "doc_date" not in st.session_state:
        st.session_state.doc_date = date.today().strftime("%Y-%m-%d")
    if "category" not in st.session_state:
        st.session_state.category = "Admin"
    if "desc" not in st.session_state:
        st.session_state.desc = ""

    # Form for document upload
    with st.form(key="upload_form", clear_on_submit=True, border=False):
        # Upload file
        uploaded_file = st.file_uploader("Select a **file to upload**:", type=["pdf", "txt", "md"])

        # Form fields
        c1, c2 = st.columns([1, 1])
        with c1:
            doc_date = st.date_input(
                "**Publish Date**:",
                min_value=date(2024, 6, 1),
                value=st.session_state.doc_date
            )
        with c2:
            category = st.selectbox(
                "**File Category**:",
                options=["Admin", "Contract", "Evidence", "Meetings", "Policy", "Prescribing", "Research", "Staff", "Targets"],
                index=["Admin", "Contract", "Evidence", "Meetings", "Policy", "Prescribing", "Research", "Staff", "Targets"].index(st.session_state.category),
            )

        desc = st.text_input("Brief **Document Description**:", value=st.session_state.desc)

        # Form buttons
        submit_button = st.form_submit_button(label=":material/cloud_upload: Upload Document")


    if submit_button and uploaded_file is not None and desc:
        # File processing logic
        raw_doc = ""

        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    raw_doc += page.extract_text()

        elif uploaded_file.type == "text/plain" or uploaded_file.type == "text/markdown":
            raw_doc = uploaded_file.read().decode("utf-8")



        if st.session_state['developer_mode']:
            raw_doc = raw_doc

       # Sidebar view of extracted text
        with st.spinner("Text pre-processing..."):
            text_doc = clean_text(raw_doc)
            time.sleep(1)  # Add delay to ensure spinner is visible


        # Display extracted text only in developer mode
        if st.session_state["developer_mode"]:
            with st.expander(":material/match_word: Extracted **Text**"):
                token_count = count_tokens_in_string(text_doc)
                st.write(f"### 🚥 Token in document: {token_count}")
                st.text(text_doc)

        # Semantic chunking
        with st.spinner("Semantic Chunking..."):
            documents = prepare_documents_with_semantic_chunker(
                text_doc, desc, uploaded_file.name, uploaded_file.size, category, str(doc_date)
            )

        # Display chunked text only in developer mode
        if st.session_state["developer_mode"]:
            with st.expander(":material/grid_view: Chunked **Text**"):
                st.json(documents)

        # Developer Mode: Handle "Upload to Pinecone" button
        if st.session_state["developer_mode"]:
            # DO TESTS ON DOCUMENTS - HOW MANY CHUNKS AND IF ANY TEXT > 7000 WORDS TIKTOKEN CHECK
            if st.button(":material/reset_settings: Reset App"):
                # Reset form fields and states
                st.session_state.uploaded_file = None
                st.session_state.doc_date = date.today().strftime("%Y-%m-%d")
                st.session_state.category = "Admin"
                st.session_state.desc = ""
                st.rerun()

        else:

            # Upload to Pinecone
            with st.spinner("Uploading to Pinecone Vector DB..."):
                total_documents = len(documents)
                progress_bar = st.progress(0, text="Uploading to Pinecone Vector Database...")
                for i, document in enumerate(documents):
                    doc_id = str(uuid.uuid4())
                    vector_store.add_texts(
                        texts=[document["text"]],
                        metadatas=[document["metadata"]],
                        ids=[doc_id],
                    )
                    progress_bar.progress((i + 1) / total_documents)

            # Log document in Google Sheet
            with st.spinner("Log Document in Google Sheet..."):
                add_doc_googlesheet(uploaded_file.name, desc, uploaded_file.size, category, doc_date)
                st.success("✅ Upload complete!")
                time.sleep(5)

            # Reset form fields and states
            st.session_state.uploaded_file = None
            st.session_state.doc_date = date.today().strftime("%Y-%m-%d")
            st.session_state.category = "Admin"
            st.session_state.desc = ""
            st.rerun()



# Tab: Manage Knowledge
with tabs[2]:
    st.image("images/header.png")
    st.header(":material/school: Knowledge")
    st.caption("**Practice Pilot** aggregates knowledge from a range of authoritative sources, including clinical decision support content, medical literature, government reports, practice management resources, and patient education materials, meeting notes, with new updates and additions made continuously since launch.")
    data = conn.read(

        worksheet="Sheet1",
        ttl="5",
    )
    data['Publish Date'] = pd.to_datetime(data['Publish Date'])


    # Set 'Publish Date' as the index
    data.set_index('Publish Date', inplace=True)

    # Resample the data weekly and aggregate file size
    weekly_data = data.resample("D").agg({"File Size": "sum"})

    # Reset the index for plotting
    weekly_data.reset_index(inplace=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(x="Publish Date", y="File Size", data=weekly_data, color="#53b3c5", linewidth=2)

    # Customize the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#888888")
    plt.xlabel("Date")
    plt.ylabel("File Size (sum)")
    plt.title("Data Uploaded to Vector DB")
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Example: Streamlit multiselect for category selection
    selector = st.multiselect(
        "Filter knowledge **by Category**:",
        options=["Admin", "Contract", "Evidence", "Meetings", "Policy", "Prescribing", "Research", "Staff", "Targets"],
        default=["Admin", "Contract", "Evidence", "Meetings", "Policy", "Prescribing", "Research", "Staff", "Targets"],
    )

    st.container(height=15, border=False)
    # Filter the dataframe based on selected categories
    if "Category" in data.columns:
        filtered_data = data[data["Category"].isin(selector)]
    else:
        st.warning("The 'Category' column is missing in the dataframe.")
        filtered_data = data

    # Display the filtered dataframe
    st.dataframe(filtered_data, height=800)



# Tab: About
with tabs[3]:
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
### The Power of RAG and Embeddings in Practice Pilot
By integrating Retrieval-Augmented Generation with our Pinecone Vector database, we're able to provide an unparalleled search experience for healthcare professionals. Our system can generate contextualized answers, recall relevant information from large databases, and even detect biases in medical literature. The combination of the retrieval-augmented generation approach and cosine search enables us to deliver accurate, informative results that go beyond traditional search engines. This is made possible by the power of embeddings, which enable our system to capture subtle patterns and relationships within the data, ultimately enhancing the user experience for healthcare professionals.
### Chatting with Practice Pilot: Tips for Success
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
