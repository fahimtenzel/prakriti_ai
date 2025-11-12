import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

EMB_MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, vector_dir="vectorstore"):
        self.vector_dir = vector_dir
        self.model = SentenceTransformer(EMB_MODEL_NAME)
        idx_path = os.path.join(vector_dir, "faiss.index")
        meta_path = os.path.join(vector_dir, "metadatas.pkl")
        emb_path = os.path.join(vector_dir, "embeddings.pkl")
        if not os.path.exists(idx_path):
            raise FileNotFoundError("No vectorstore found. Ingest docs first.")
        self.index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            self.metadatas = pickle.load(f)
        with open(emb_path, "rb") as f:
            self.embeddings = pickle.load(f)

    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query]).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            md = self.metadatas[idx]
            emb = self.embeddings[idx]
            results.append({"metadata": md, "embedding": emb})
        return results
