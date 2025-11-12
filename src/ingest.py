import os
import pdfplumber
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss, pickle
import numpy as np
from src.utils import clean_text, get_text_chunks

EMB_MODEL_NAME = "all-MiniLM-L6-v2"

class Ingestor:
    def __init__(self, vector_dir="vectorstore"):
        self.model = SentenceTransformer(EMB_MODEL_NAME)
        self.vector_dir = vector_dir
        os.makedirs(self.vector_dir, exist_ok=True)
        self._load_index_if_exists()

    def _load_index_if_exists(self):
        idx_path = os.path.join(self.vector_dir, "faiss.index")
        meta_path = os.path.join(self.vector_dir, "metadatas.pkl")
        emb_path = os.path.join(self.vector_dir, "embeddings.npy")
        if os.path.exists(idx_path) and os.path.exists(meta_path) and os.path.exists(emb_path):
            self.index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as f:
                self.metadatas = pickle.load(f)
            self.embeddings = np.load(emb_path)
        else:
            self.index = None
            self.metadatas = []
            self.embeddings = None

    def save_index(self):
        idx_path = os.path.join(self.vector_dir, "faiss.index")
        meta_path = os.path.join(self.vector_dir, "metadatas.pkl")
        emb_path = os.path.join(self.vector_dir, "embeddings.npy")
        faiss.write_index(self.index, idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)
        np.save(emb_path, self.embeddings)

    def ingest_pdf(self, file_path, source_name=None):
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                txt = page.extract_text() or ""
                txt = clean_text(txt)
                if txt:
                    texts.append({"text": txt, "page": i+1})
        self._ingest_text_blocks(texts, source=source_name or os.path.basename(file_path))

    def ingest_url(self, url):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch URL {url}: {e}")
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = clean_text(text)
        self._ingest_text_blocks([{"text": text, "page": 1}], source=url)

    def _ingest_text_blocks(self, blocks, source="<unknown>"):
        all_chunks, metadatas = [], []
        for b in blocks:
            chunks = get_text_chunks(b["text"])
            for i, c in enumerate(chunks):
                meta = {"source": source, "page": b.get("page", 1), "chunk_index": i, "text": c}
                metadatas.append(meta)
                all_chunks.append(c)

        if not all_chunks:
            return

        embs = self.model.encode(all_chunks, show_progress_bar=True)
        embs = np.array(embs, dtype="float32")

        d = embs.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(d)
            self.index.add(embs)
            self.embeddings = embs
            self.metadatas = metadatas
        else:
            self.index.add(embs)
            self.embeddings = np.vstack([self.embeddings, embs])
            self.metadatas.extend(metadatas)

        self.save_index()
