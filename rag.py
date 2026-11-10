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

embeddings = embedder.encode(documents)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


def retrieve_context(query):

    query_embedding = embedder.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        k=1
    )

    return documents[indices[0][0]]
