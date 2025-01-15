import streamlit as st
import pdfplumber
import time
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from semantic_chunker import prepare_documents_with_semantic_chunker
from pinecone_module import upload_documents_to_pinecone
from query_module import query_vector_database, generate_augmented_response
import uuid

# Streamlit settings
st.set_page_config(page_title="PracticePilot")
st.logo("images/title.png", size='large')

# Set API keys securely
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)


st.sidebar.title(':material/settings: System Settings')

# Reload app button
clear_button = st.sidebar.button("Reload App / Upload another document")
if clear_button:
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# Tab layout
tabs = st.tabs([":material/robot_2: Chat", ":material/upload: Upload Documents", ":material/school: Manage Knowledge", ":material/privacy_tip: About"])

# Tab: Chat
with tabs[0]:
    st.header(":material/robot_2: Chat with PracticePilot")

    index_name = "practicepilot"
    embed_model = "text-embedding-ada-002"
    top_k = st.sidebar.number_input("**top_k** value:", min_value=5, max_value=10, help="Specify how many vectors are returned.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    messages_container = st.container()
    with messages_container:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    st.markdown("<div style='position: fixed; bottom: 0; width: 100%;'>", unsafe_allow_html=True)
    user_input = st.chat_input("Ask me a question about Primary Care?")
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
    st.image("images/header.png")
    st.header(":material/upload: Upload Documents", help="Upload new documents to the Vector Database to extend the app's knowledge.")
    pc = Pinecone(api_key=pinecone_api_key)

    if index_name not in [index["name"] for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model=embed_model)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    choice = st.radio("Select document type:", options=["PDF", "TXT"], index=0, horizontal=True)

    uploaded_file = st.file_uploader(f"Upload a {choice} file.", type=choice.lower())
    if uploaded_file is not None:
        desc = st.text_input("Document Description:", placeholder="Enter document description.")
        if desc:
            text_doc = ""

            if choice == "PDF":
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        text_doc += page.extract_text()

            elif choice == "TXT":
                text_doc = uploaded_file.read().decode("utf-8")

            with st.sidebar.expander(":material/menu_book: **Extracted Text**"):
                st.text(text_doc)

            with st.spinner("Processing document..."):
                documents = prepare_documents_with_semantic_chunker(text_doc, desc, uploaded_file.name, uploaded_file.size)

            with st.spinner("Uploading to Pinecone..."):
                total_documents = len(documents)
                progress_bar = st.progress(0)
                for i, document in enumerate(documents):
                    doc_id = str(uuid.uuid4())
                    vector_store.add_texts(
                        texts=[document["text"]],
                        metadatas=[document["metadata"]],
                        ids=[doc_id],
                    )
                    progress_bar.progress((i + 1) / total_documents)

            st.success("✅ Document successfully uploaded!")

# Tab: Manage Knowledge
with tabs[2]:
    st.image("images/header.png")
    st.header(":material/school: Manage Knowledge")
    st.write("Manage the knowledge store in your vector database.")

# Tab: About
with tabs[3]:
    st.image("images/header.png")
    st.header(":material/privacy_tip: About")
    st.write("The tech behind this app.")
    st.markdown("[RAGatouille](https://github.com/AnswerDotAI/RAGatouille)")
    st.write("Available secrets:", list(st.secrets.keys()))
