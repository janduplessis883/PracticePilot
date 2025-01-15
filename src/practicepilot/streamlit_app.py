import streamlit as st
import pdfplumber
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import uuid

from semantic_chunker import prepare_documents_with_semantic_chunker
from pinecone_module import upload_documents_to_pinecone

desc = ''

st.set_page_config(page_title="PracticePilot")
st.logo(
    "images/logo_side.png",
    link="https://github.com/janduplessis883/PracticePilot/tree/master",
    size="large",
)


st.sidebar.title(':material/settings: System Settings')


# Create tabs
tabs = st.tabs([":material/robot_2: Chat", ":material/upload: Upload Documents", ":material/school: Manage Knowledge", ":material/privacy_tip: About"])

# Tab: Chat
with tabs[0]:
    st.header(":material/robot_2: Chat with PracticePilot")
    st.write(":material/robot_2: **Chat** with **PracticePilot** re Primary Care knowledge.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Create a container for chat messages
    messages_container = st.container(height=450, border=True)

    # Display chat messages inside the container
    with messages_container:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Message input field at the bottom
    st.markdown("<div style='position: fixed; bottom: 0; width: 100%;'>", unsafe_allow_html=True)
    user_input = st.chat_input("Enter your message:")
    st.markdown("</div>", unsafe_allow_html=True)

    # Process user input and update the chat
    if user_input:
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with messages_container:
            with st.chat_message("user", avatar="images/user.png"):
                st.markdown(user_input)

        # Simulate bot response
        bot_response = "This is a simulated response!"
        st.session_state["messages"].append({"role": "assistant", "content": bot_response})
        with messages_container:
            with st.chat_message("assistant", avatar="images/avatar.png"):
                st.markdown(bot_response)

# Tab: Upload Documents
with tabs[1]:
    st.image("images/header.png")
    #pinecone_api_key = st.secrets['PINECONE_API_KEY']
    pc = Pinecone(api_key="pcsk_3yD3bu_R8mZx94Thw4S8kVVnYzYZQmoAsppttSv7EP7nxPuUK5H5vQgQN1TPuadzB5UBrT")

    # Create Vector database if it does not exist
    index_name = "practicepilot"  # Change this if needed

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Initialize the vector store
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)


    st.header(":material/upload: Upload Documents", help="Upload new documents to the Vector Database to extend the apps knowledge.")

    choice = st.radio("Select document type:", options=["PDF", "TXT"], index=0, horizontal=True)

    if choice == "PDF":

        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Display the document description input only after a file is uploaded
            desc = st.text_input("Document Description:", placeholder="Enter document description.")
            st.divider()
            # Proceed only if both a file is uploaded and a description is provided
            if desc != "":
                text_doc = ""

                # Use pdfplumber to extract text
                with pdfplumber.open(uploaded_file) as pdf:
                    # Extract text from each page
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        text_doc += text

                    with st.expander(":material/menu_book: **Extracted Text**"):
                        st.write(text_doc)

                with st.spinner("Semantic Splitting of doc..."):
                    with st.expander(":material/vertical_split:  **Semantic Chunks**"):
                        # Prepare the documents with semantic chunking
                        documents = prepare_documents_with_semantic_chunker(
                            text_doc, desc, uploaded_file.name, uploaded_file.size
                        )
                        st.json(documents)

                with st.spinner("Uploading embeddings to Pinecone..."):
                    # Upload the documents to Pinecone
                    upload_documents_to_pinecone(documents, vector_store)


    elif choice == "TXT":

        uploaded_file = st.file_uploader("Upload a TXT file.", type="txt")

        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Display the document description input after a file is uploaded
            desc = st.text_input("Document Description:", placeholder="Enter document description.")
            st.divider()
            # Proceed only if both the file is uploaded and description is provided
            if desc != "":
                # Read and display the uploaded file
                text_doc = uploaded_file.read().decode("utf-8")
                with st.expander(":material/menu_book: **Extracted Text**"):
                    st.text(text_doc)

                with st.spinner("Semantic Splitting of doc..."):
                    with st.expander(":material/vertical_split:  **Semantic Chunks**"):
                        # Prepare the documents with semantic chunking
                        documents = prepare_documents_with_semantic_chunker(text_doc, desc, uploaded_file.name, uploaded_file.size)
                        st.json(documents)

                with st.spinner("Uploading embeddings to Pinecone..."):
                    # Upload the documents to Pinecone
                    upload_documents_to_pinecone(documents, vector_store)










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
    st.markdown("[RAGatouille](https://github.com/AnswerDotAI/RAGatouille?ref=dailydoseofds.com)")
