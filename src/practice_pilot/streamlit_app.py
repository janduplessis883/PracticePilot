import streamlit as st
import pinecone

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
    st.header("Chat")
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
    st.header("Upload Documents")
    st.write("This is the Upload Documents tab where users can upload files.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        # Display file details
        st.write("**File Details:**")
        st.write(f"Name: {uploaded_file.name}")
        st.write(f"Type: {uploaded_file.type}")
        st.write(f"Size: {uploaded_file.size} bytes")

        # Optionally read the content (e.g., for text files)
        if uploaded_file.type in ["text/plain", "application/pdf"]:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            st.text_area("File Content", content, height=200)

# Tab: Manage Knowledge
with tabs[2]:
    st.header("Manage Knowledge")
    st.write("Manage the knowledge store in your vector database.")


# Tab: About
with tabs[3]:
    st.header(":material/privacy_tip: About")
    st.write("The tech behind this app.")
    st.markdown("[RAGatouille](https://github.com/AnswerDotAI/RAGatouille?ref=dailydoseofds.com)")
