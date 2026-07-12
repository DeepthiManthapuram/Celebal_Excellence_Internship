import streamlit as st

from retriever import (
    load_embeddings,
    load_vector_store,
    retrieve_documents,
    build_context,
    build_prompt,
)

from generator import generate_answer


st.set_page_config(
    page_title="RAG Document Question Answering",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG Document Question Answering System")

st.write(
    "Ask questions about the documents stored in the FAISS vector database."
)

# Load FAISS only once
@st.cache_resource
def load_rag():

    embeddings = load_embeddings()

    vector_db = load_vector_store(embeddings)

    return vector_db


vector_db = load_rag()

question = st.text_input("Enter your question")

if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching documents..."):

        results = retrieve_documents(vector_db, question)

        context = build_context(results)

        prompt = build_prompt(context, question)

    with st.spinner("Generating answer using Gemini..."):

        answer = generate_answer(prompt)

    st.success("Answer Generated")

    st.subheader("Answer")

    st.write(answer)

    with st.expander("Retrieved Context"):

        st.text(context)