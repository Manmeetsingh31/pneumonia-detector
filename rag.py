import os
from pathlib import Path

import faiss
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
KB_FOLDER = BASE_DIR / "knowledge_base"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource(show_spinner="Loading the RAG embedding model...")
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


embedder = load_embedder()


# ==========================================
# 3. LOAD AND CHUNK KNOWLEDGE BASE
# ==========================================

def chunk_text(text, chunk_size=120, overlap=20):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks


if not KB_FOLDER.exists():
    raise FileNotFoundError("knowledge_base folder not found!")


documents = []
metadata = []

for file_name in sorted(os.listdir(KB_FOLDER)):
    if file_name.lower().endswith(".txt"):
        file_path = KB_FOLDER / file_name
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

        if not content:
            continue

        for chunk_number, chunk in enumerate(chunk_text(content), start=1):
            documents.append(chunk)
            metadata.append({"source": file_name, "chunk": chunk_number})

if not documents:
    raise ValueError("No text documents found in knowledge_base!")


# ==========================================
# 4. CREATE FAISS VECTOR INDEX
# ==========================================

@st.cache_resource(show_spinner="Building the knowledge-base search index...")
def build_faiss_index(texts):
    vectors = embedder.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    vector_index = faiss.IndexFlatIP(vectors.shape[1])
    vector_index.add(vectors)
    return vector_index


index = build_faiss_index(tuple(documents))


# ==========================================
# 5. RETRIEVE RELEVANT CONTEXT
# ==========================================

def retrieve_context(query, top_k=3):
    query_vector = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(documents))
    scores, indices = index.search(query_vector, k)

    results = []
    for score, doc_index in zip(scores[0], indices[0]):
        if doc_index < 0:
            continue
        results.append({
            "text": documents[doc_index],
            "source": metadata[doc_index]["source"],
            "chunk": metadata[doc_index]["chunk"],
            "similarity": float(score),
        })

    return results


# ==========================================
# 6. GENERATE ANSWER USING GROQ CLOUD
# ==========================================

def generate_answer(question, retrieved_results):
    if not retrieved_results:
        return "I could not find relevant information in the knowledge base."

    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it under App settings → Secrets "
            "in Streamlit Community Cloud."
        )

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in retrieved_results
    )

    system_prompt = (
        "You are a helpful educational assistant for a Pneumonia Detection project. "
        "Answer using only the supplied knowledge-base context. If the answer is not "
        "available in that context, clearly say the knowledge base does not contain "
        "enough information. Do not invent medical facts. Do not diagnose patients "
        "or prescribe treatment. Remind users to consult qualified healthcare "
        "professionals for medical decisions."
    )

    user_prompt = f"KNOWLEDGE-BASE CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"

    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_completion_tokens": 800,
        },
        timeout=90,
    )

    if response.status_code == 401:
        raise RuntimeError("Groq rejected the API key. Check GROQ_API_KEY in Streamlit Secrets.")
    if response.status_code == 429:
        raise RuntimeError("Groq rate limit or quota reached. Please wait and try again.")

    response.raise_for_status()
    payload = response.json()

    try:
        return payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("Groq returned an unexpected response format.") from error


# ==========================================
# 7. COMPLETE RAG PIPELINE
# ==========================================

def ask_question(question, top_k=3):
    retrieved_results = retrieve_context(question, top_k=top_k)
    answer = generate_answer(question, retrieved_results)
    return answer, retrieved_results


if __name__ == "__main__":
    test_question = "What are the common symptoms of pneumonia?"
    test_answer, test_sources = ask_question(test_question)
    print(test_answer)
    print("\nRetrieved sources:")
    for source in test_sources:
        print(
            f"- {source['source']} | chunk {source['chunk']} "
            f"| similarity {source['similarity']:.4f}"
        )
