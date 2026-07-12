import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from generator import generate_answer


# ----------------------------------------------------
# Logging Configuration
# ----------------------------------------------------
logging.basicConfig(
    filename="rag.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ----------------------------------------------------
# Load Embedding Model
# ----------------------------------------------------
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# ----------------------------------------------------
# Load FAISS Vector Store
# ----------------------------------------------------
def load_vector_store(embeddings):

    vector_db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db


# ----------------------------------------------------
# Accept User Question
# ----------------------------------------------------
def get_user_question():

    question = input("\nEnter your question: ")

    return question


# ----------------------------------------------------
# Retrieve Documents
# ----------------------------------------------------
def retrieve_documents(vector_db, question):

    # Retrieve top 10 similar chunks
    results = vector_db.similarity_search(
        question,
        k=10
    )

    filtered_results = []

    question_lower = question.lower()

    # Resume related questions
    if "resume" in question_lower:

        for doc in results:

            source = doc.metadata.get("source", "").lower()

            if "resume" in source:
                filtered_results.append(doc)

    # Machine Learning related questions
    elif "machine learning" in question_lower:

        for doc in results:

            source = doc.metadata.get("source", "").lower()

            if "notes" in source:
                filtered_results.append(doc)

    # Python related questions
    elif "python" in question_lower:

        for doc in results:

            source = doc.metadata.get("source", "").lower()

            if "python_notes" in source:
                filtered_results.append(doc)

    else:

        filtered_results = results

    # Fallback
    if len(filtered_results) == 0:

        filtered_results = results

    return filtered_results[:3]


# ----------------------------------------------------
# Build Context
# ----------------------------------------------------
def build_context(results):

    context = ""

    for i, doc in enumerate(results, start=1):

        context += f"Chunk {i}\n"

        context += doc.page_content

        context += "\n\n"

    return context


# ----------------------------------------------------
# Build Prompt
# ----------------------------------------------------
def build_prompt(context, question):

    prompt = f"""
You are a Retrieval-Augmented Generation (RAG) assistant.

Use ONLY the information provided in the Context.

Rules:

1. Do NOT use outside knowledge.
2. Do NOT make up information.
3. If the question asks for a list, return only the list.
4. Do NOT explain unless explicitly asked.
5. If the answer is not present in the context, reply:

"I couldn't find the answer in the provided documents."

Context:

{context}

Question:

{question}

Answer:
"""

    return prompt


# ----------------------------------------------------
# Main
# ----------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("RAG DOCUMENT QUESTION ANSWERING SYSTEM")
    print("=" * 60)

    # Load Embedding Model
    embeddings = load_embeddings()

    # Load Vector Store
    vector_db = load_vector_store(embeddings)

    print("\nFAISS Vector Database Loaded Successfully!")

    logging.info("FAISS Vector Database Loaded Successfully")

    # Accept Question
    question = get_user_question()

    logging.info(f"User Question : {question}")

    # Retrieve Documents
    results = retrieve_documents(vector_db, question)

    logging.info(f"Retrieved {len(results)} chunks")

    print("\n")
    print("=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)

    for i, doc in enumerate(results, start=1):

        print(f"\nChunk {i}")
        print("-" * 40)

        print("Source :", doc.metadata.get("source"))

        print()

        print(doc.page_content)

    # Build Context
    context = build_context(results)

    # Build Prompt
    prompt = build_prompt(context, question)

    print("\n")
    print("=" * 60)
    print("PROMPT SENT TO GEMINI")
    print("=" * 60)

    print(prompt)

    print("\nGenerating Answer...\n")

    # Generate Answer
    answer = generate_answer(prompt)

    logging.info("Answer Generated Successfully")

    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(answer)

    logging.info("Program Executed Successfully")