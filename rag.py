
import os
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
KB_FOLDER = BASE_DIR / "knowledge_base"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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

print("FAISS index created successfully!")

print("Total documents:", len(documents))
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
# 7. TEST RETRIEVAL
# ==========================================

if __name__ == "__main__":

    question = "What are the common symptoms of pneumonia?"

    print("\nUSER QUESTION:")
    print(question)

    results = retrieve_context(question)

    print("\nRETRIEVED DOCUMENTS:")

    for result in results:

        print("\nSource:", result["source"])
        print("Chunk:", result["chunk"])
        print("Similarity:", round(result["similarity"], 4))

        print("Content:")
        print(result["text"])

    print("\nRETRIEVAL TEST COMPLETED!")

