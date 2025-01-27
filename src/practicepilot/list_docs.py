from streamlit_gsheets import GSheetsConnection
import streamlit as st
import pandas as pd

import weave

# Initialize the Streamlit GSheets connection
@weave.op()
def add_doc_googlesheet(file_name, doc_desc, file_size, category, pub_date):
    """
    Appends a row of document information to a Google Sheet using Streamlit's GSheetsConnection.

    Args:
        file_name (str): Name of the file being added.
        doc_desc (str): Description of the document.
        file_size (str): Size of the file.
        category (str): Category of the document.
        pub_date (str): Publication date of the document.

    Returns:
        bool: True if the row is successfully appended, False otherwise.
    """
    try:
        gsheets = GSheetsConnection(...)
        conn = st.connection("gsheets", type=GSheetsConnection)
        existing_data = conn.read(worksheet="Sheet1", ttl=5)
        # Prepare the row data
        new_data = pd.DataFrame(
                [
                    {
                        "Category": category,
                        "Filename": file_name,
                        "File Description": doc_desc,
                        "Publish Date": pub_date,
                        "File Size": file_size,
                    }
                ]
            )

        updated_df = pd.concat([existing_data, new_data], ignore_index=True)
        # Update Google Sheets with the new vendor data
        conn.update(worksheet="Sheet1", data=updated_df)
        return True
    except Exception as e:
        # Log the error for debugging purposes
        st.error(f"Error appending row to Google Sheet: {e}")
        return False
