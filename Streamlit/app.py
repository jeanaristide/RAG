import os
import streamlit as st

DEFAULT_TERM_STR = (
    "Make a list of terms and definitions that are defined in the context, "
    "with one pair on each line. "
    "If a term is missing it's definition, use your best judgment. "
    "Write each line as as follows:\nTerm: <term> Definition: <definition>"
)

st.title("🦙 Llama Index Term Extractor 🦙")

setup_tab, upload_tab = st.tabs(["Setup", "Upload/Extract Terms"])

with setup_tab:
    st.subheader("LLM Setup")
    api_key = st.text_input("Enter your OpenAI API key here", type="password")
    llm_name = st.selectbox(
        "Which LLM?", ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]
    )
    model_temperature = st.slider(
        "LLM Temperature", min_value=0.0, max_value=1.0, step=0.1
    )
    term_extract_str = st.text_area(
        "The query to extract terms and definitions with.",
        value=DEFAULT_TERM_STR,
    )

with upload_tab:
    st.subheader("Extract and Query Definitions")
    document_text = st.text_area("Or enter raw text")
    if st.button("Extract Terms and Definitions") and document_text:
        with st.spinner("Extracting..."):
            extracted_terms = document_text  # this is a placeholder!
        st.write(extracted_terms)