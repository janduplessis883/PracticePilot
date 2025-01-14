import streamlit as st


page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://github.com/janduplessis883/PracticePilot/blob/master/images/bg.png?raw=true");
  background-size: cover;
}
</style>
"""

st.markdown(page_element, unsafe_allow_html=True)
