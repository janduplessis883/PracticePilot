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
st.image("images/header.png")

# Create tabs
tabs = st.tabs([":material/robot_2: Chat", ":material/upload: Upload Documents", ":material/school: Manage Knowledge", ":material/privacy_tip: About"])

# Tab: Chat
with tabs[0]:
    st.header(":material/robot_2: Chat")
    st.write("This is the Chat tab where users can interact.")

    # Chat simulation
    user_input = st.text_input("Enter your message:", "")
    if st.button("Send"):
        if user_input:
            st.write(f"**You:** {user_input}")
            st.write("**Bot:** This is a simulated response!")
        else:
            st.warning("Please enter a message to send.")

# Tab: Upload Documents
with tabs[1]:
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
    st.header(":material/school: Manage Knowledge")
    st.write("Manage the knowledge store in your vector database.")













# Tab: About
with tabs[3]:
    st.header(":material/privacy_tip: About")
    st.write("The tech behind this app.")
    st.markdown("[RAGatouille](https://github.com/AnswerDotAI/RAGatouille?ref=dailydoseofds.com)")
