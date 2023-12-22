import streamlit as st

st.title("🦙 Llama Index Term Extractor 🦙")

document_text = st.text_area("Or enter raw text")
if st.button("Extract Terms and Definitions") and document_text:
    with st.spinner("Extracting..."):
        extracted_terms = document_text  # this is a placeholder!
    st.write(extracted_terms)