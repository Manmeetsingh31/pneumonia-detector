from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Load embedding model
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = []
doc_names = []

kb_folder = "knowledge_base"

# Load all txt files
for file in os.listdir(kb_folder):

    if file.endswith(".txt"):

        with open(
            os.path.join(kb_folder, file),
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

        documents.append(text)
        doc_names.append(file)

# Create embeddings
embeddings = embedder.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    np.array(embeddings).astype("float32")
)


def retrieve_context(query):

    query_embedding = embedder.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k=1
    )

    return documents[
        indices[0][0]
    ]


# TEST CODE

print("=" * 50)
print("TESTING RAG SYSTEM")
print("=" * 50)

result = retrieve_context(
    "Tell me about pneumonia"
)

print(result)
