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

from semantic_chunker import prepare_documents_with_semantic_chunker
from pinecone_module import upload_documents_to_pinecone
from query_module import query_vector_database, generate_augmented_response
from list_docs import add_doc_googlesheet

st.set_page_config(page_title="PracticePilot")
st.logo("images/title.png", size='large')

# Streamlit settings


# Set API keys securely
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

gsheets = GSheetsConnection(...)
conn = st.connection("gsheets", type=GSheetsConnection)


# Reload app button
clear_button = st.sidebar.button("Clear Forms / Reset App")
if clear_button:
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

st.sidebar.title(':material/chat: Chat Settings')

# Tab layout
tabs = st.tabs([":material/robot_2: Chat", ":material/upload: Upload Documents", ":material/school: Manage Knowledge", ":material/privacy_tip: About"])

# Tab: Chat
with tabs[0]:
    st.header(":material/robot_2: Chat with PracticePilot")

    index_name = "practicepilot"
    embed_model = "text-embedding-ada-002"
    filter_date = st.sidebar.date_input("Only consider knowledge **after**:", value="2024-06-01", format="YYYY-MM-DD")
    top_k = st.sidebar.number_input("**top_k** - how many vectors to return:", min_value=5, max_value=10, help="Specify how many vectors are returned.")

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
    # Main interface
    st.image("images/header.png")
    st.header(":material/upload: Upload Documents", help="Upload new documents to the Vector Database to extend the app's knowledge.")
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
            time.sleep(1)

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
    with st.form(key="upload_form", clear_on_submit=True):
        # Upload file
        uploaded_file = st.file_uploader("Select a file to upload:", type=["pdf", "txt"])

        # Form fields
        c1, c2 = st.columns([1, 1])
        with c1:
            doc_date = st.date_input(
                "Doc Publish Date:",
                min_value=date(2024, 6, 1),
                value=st.session_state.doc_date
            )
        with c2:
            category = st.selectbox(
                "File Category:",
                options=["Admin", "Contract", "Evidence", "Policy", "Prescribing", "Research", "Staff", "Targets"],
                index=["Admin", "Contract", "Evidence", "Policy", "Prescribing", "Research", "Staff", "Targets"].index(st.session_state.category),
            )
        desc = st.text_input("Document Description:", value=st.session_state.desc)

        # Form buttons
        submit_button = st.form_submit_button(label="Upload Document")


    if submit_button and uploaded_file is not None and desc:
        # File processing logic
        text_doc = ""

        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text_doc += page.extract_text()

        elif uploaded_file.type == "text/plain":
            text_doc = uploaded_file.read().decode("utf-8")

        # Sidebar view of extracted text
        with st.sidebar.expander(":material/menu_book: **Extracted Text**"):
            st.text(text_doc)

        # Processing document and updating Google Sheet
        with st.spinner("Processing document & Writing G Sheet..."):
            documents = prepare_documents_with_semantic_chunker(
                text_doc, desc, uploaded_file.name, uploaded_file.size, category, str(doc_date)
            )
            add_doc_googlesheet(uploaded_file.name, desc, uploaded_file.size, category, doc_date)

        # Uploading to Pinecone
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

        # Success message and reset
        st.success("✅ Document successfully uploaded!")
        time.sleep(2)

        # Reset form fields
        st.session_state.uploaded_file = None
        st.session_state.doc_date = date(2024, 6, 1)
        st.session_state.category = "Admin"
        st.session_state.desc = ""
        st.rerun()



# Tab: Manage Knowledge
with tabs[2]:
    st.image("images/header.png")
    st.header(":material/school: Manage Knowledge")
    st.write("Manage the knowledge store in your vector database.")
    data = conn.read(

        worksheet="Sheet1",
        ttl="5",
    )
    st.dataframe(data)


# Tab: About
with tabs[3]:
    st.image("images/header.png")
    st.header(":material/privacy_tip: About")
    st.write("The tech behind this app.")
    st.markdown("[RAGatouille](https://github.com/AnswerDotAI/RAGatouille)")
    st.write("Available secrets:", list(st.secrets.keys()))
