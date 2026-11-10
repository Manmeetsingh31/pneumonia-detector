
import os
from pathlib import Path

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
KB_FOLDER = BASE_DIR / "knowledge_base"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

OLLAMA_API = "http://localhost:11434/api/generate"


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

embedder = SentenceTransformer(MODEL_NAME)


# ==========================================
# 3. LOAD KNOWLEDGE BASE
# ==========================================

documents = []
metadata = []


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


print("Reading knowledge base...")

if not KB_FOLDER.exists():
    raise FileNotFoundError("knowledge_base folder not found!")

for file in sorted(os.listdir(KB_FOLDER)):

    if file.endswith(".txt"):

        path = KB_FOLDER / file

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            continue

        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):

            documents.append(chunk)

            metadata.append({
                "source": file,
                "chunk": i + 1
            })


if not documents:
    raise ValueError("No text documents found in knowledge_base!")


# ==========================================
# 4. GENERATE EMBEDDINGS
# ==========================================

print("Generating embeddings...")

embeddings = embedder.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = embeddings.astype("float32")


# ==========================================
# 5. CREATE FAISS VECTOR INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("\nFAISS index created successfully!")

print("Total chunks:", index.ntotal)
print("Vector dimensions:", dimension)


# ==========================================
# 6. RETRIEVE RELEVANT CONTEXT
# ==========================================

def retrieve_context(query, top_k=3):

    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    k = min(top_k, len(documents))

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        results.append({
            "text": documents[idx],
            "source": metadata[idx]["source"],
            "chunk": metadata[idx]["chunk"],
            "similarity": float(score)
        })

    return results


# ==========================================
# 7. GENERATE ANSWER USING OLLAMA
# ==========================================

def generate_answer(question, retrieved_results):

    if not retrieved_results:
        return "I could not find relevant information in the knowledge base."

    # Combine retrieved chunks into context
    context = "\n\n".join(
        [
            f"Source: {item['source']}\n{item['text']}"
            for item in retrieved_results
        ]
    )

    # Build the prompt
    prompt = f"""
You are a helpful educational assistant for a
Pneumonia Detection project.

Answer the user's question using ONLY the
provided context.

If the answer is not available in the context,
clearly say that the knowledge base does not
contain enough information.

Do not invent medical facts.
Do not diagnose patients or prescribe treatment.
Remind users that medical questions should be
discussed with qualified healthcare professionals
when appropriate.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    # Send prompt to local Ollama API
    response = requests.post(
        OLLAMA_API,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    return result["response"].strip()


# ==========================================
# 8. COMPLETE RAG PIPELINE
# ==========================================

def ask_question(question, top_k=3):

    print("\nRetrieving relevant information...")

    retrieved_results = retrieve_context(
        question,
        top_k
    )

    print("\nRETRIEVED SOURCES:")

    for item in retrieved_results:
        print(
            f"- {item['source']} "
            f"(similarity: {item['similarity']:.4f})"
        )

    print("\nGenerating answer with Llama 3.2...")

    answer = generate_answer(
        question,
        retrieved_results
    )

    return answer, retrieved_results


# ==========================================
# 9. TEST THE COMPLETE RAG PIPELINE
# ==========================================

if __name__ == "__main__":

    print("\n================================")
    print(" PNEUMONIA RAG ASSISTANT")
    print("================================")

    question = "What are the common symptoms of pneumonia?"

    print("\nUSER QUESTION:")
    print(question)

    try:

        answer, sources = ask_question(question)

        print("\n================================")
        print("GENERATED ANSWER")
        print("================================")

        print(answer)

        print("\n================================")
        print("RAG PIPELINE COMPLETED!")
        print("================================")

    except requests.exceptions.ConnectionError:

        print("\nERROR: Could not connect to Ollama.")
        print("Make sure Ollama is running.")

    except requests.exceptions.RequestException as error:

        print("\nOLLAMA API ERROR:")
        print(error)
