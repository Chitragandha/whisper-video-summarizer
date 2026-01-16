import faiss
import json
import numpy as np
import os


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    # -------------------------
    # Add embeddings
    # -------------------------
    def add(self, embeddings, metadata):
        if len(embeddings) == 0:
            return

        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    # -------------------------
    # Search
    # -------------------------
    def search(self, query_embedding, top_k=10):
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, i in enumerate(indices[0]):
            results.append((self.metadata[i], distances[0][idx]))

        return results

    # -------------------------
    # Save FAISS index + metadata
    # -------------------------
    def save(self, index_path, meta_path):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(self.index, index_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    # -------------------------
    # Load FAISS index + metadata
    # -------------------------
    def load(self, index_path, meta_path):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        self.index = faiss.read_index(index_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
