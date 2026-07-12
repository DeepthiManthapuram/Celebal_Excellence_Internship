import logging
import os

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    filename="rag.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


DATA_PATH = "data"


# -------------------------------
# Load Documents
# -------------------------------
def load_documents():

    documents = []

    for filename in os.listdir(DATA_PATH):

        filepath = os.path.join(DATA_PATH, filename)

        # PDF Files
        if filename.endswith(".pdf"):

            try:

                loader = PyPDFLoader(filepath)

                docs = loader.load()

                documents.extend(docs)

                print(f"Loaded PDF: {filename}")
                logging.info(f"Loaded PDF: {filename}")

            except Exception as e:

                print(f"Could not read PDF: {filename}")
                print(e)

                logging.error(f"Error reading PDF: {filename}")
                logging.error(e)

        # TXT Files
        elif filename.endswith(".txt"):

            try:

                loader = TextLoader(filepath, encoding="utf-8")

                docs = loader.load()

                documents.extend(docs)

                print(f"Loaded TXT: {filename}")
                logging.info(f"Loaded TXT: {filename}")

            except Exception as e:

                print(f"Could not read TXT: {filename}")
                print(e)

                logging.error(f"Error reading TXT: {filename}")
                logging.error(e)

    return documents


# -------------------------------
# Split Documents
# -------------------------------
def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=100

    )

    chunks = splitter.split_documents(documents)

    return chunks


# -------------------------------
# Create Embeddings
# -------------------------------
def create_embeddings():

    embeddings = HuggingFaceEmbeddings(

        model_name="sentence-transformers/all-MiniLM-L6-v2"

    )

    return embeddings


# -------------------------------
# Create FAISS Vector Store
# -------------------------------
def create_vector_store(chunks, embeddings):

    vector_db = FAISS.from_documents(

        documents=chunks,

        embedding=embeddings

    )

    vector_db.save_local("vectorstore")

    print("Vector Database Saved Successfully!")

    logging.info("FAISS Vector Database Created Successfully")

    return vector_db


# -------------------------------
# Generate Metrics Report
# -------------------------------
def generate_metrics(documents, chunks):

    report = f"""
===============================
RAG SYSTEM METRICS REPORT
===============================

Documents Loaded : {len(documents)}

Chunks Created : {len(chunks)}

Chunk Size : 500

Chunk Overlap : 100

Embedding Model :
sentence-transformers/all-MiniLM-L6-v2

Embedding Dimension :
384

Vector Database :
FAISS

Similarity Search :
Top K = 3

Status :
SUCCESS
"""

    with open("metrics_report.txt", "w") as file:

        file.write(report)

    logging.info("Metrics Report Generated")


# -------------------------------
# Main Function
# -------------------------------
if __name__ == "__main__":

    # Step 1
    docs = load_documents()

    print(f"\nDocuments Loaded: {len(docs)}")
    logging.info(f"Documents Loaded: {len(docs)}")

    # Step 2
    chunks = split_documents(docs)

    print(f"Chunks Created: {len(chunks)}")
    logging.info(f"Chunks Created: {len(chunks)}")

    # Step 3
    embeddings = create_embeddings()

    print("Embedding Model Loaded Successfully!")
    logging.info("Embedding Model Loaded Successfully")

    # Step 4
    vector_db = create_vector_store(chunks, embeddings)

    print("\nFAISS Vector Database Created Successfully!")
    logging.info("FAISS Vector Database Created Successfully")

    # Step 5
    generate_metrics(docs, chunks)

    print("Metrics Report Generated Successfully!")