# src/app.py
# Robust imports so Streamlit can find local modules reliably.
# Includes safe suggestion handling: clicking a suggestion sets a temporary key
# that is applied *before* the text_input widget is instantiated (avoids Streamlit error).

import os
import sys
import re

# Ensure project root and src folder are on sys.path before other imports
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Try to import Ingestor (fallback to direct file import if package import fails)
try:
    from src.ingest import Ingestor
except Exception:
    import importlib.util
    spec = importlib.util.spec_from_file_location("ingest", os.path.join(HERE, "ingest.py"))
    ingest_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ingest_mod)
    Ingestor = getattr(ingest_mod, "Ingestor")

# Import gemini client robustly
try:
    from src.gemini_client import generate_from_context
except Exception:
    try:
        from gemini_client import generate_from_context
    except Exception:
        import importlib.util
        gem_path = os.path.join(HERE, "gemini_client.py")
        if os.path.exists(gem_path):
            spec = importlib.util.spec_from_file_location("gemini_client", gem_path)
            gem_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gem_mod)
            generate_from_context = getattr(gem_mod, "generate_from_context")
        else:
            raise ModuleNotFoundError(f"Could not locate gemini_client.py at {gem_path}")

# standard imports
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss, pickle, numpy as np

# ---------------------- Helpers ----------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def strip_s_citations(text: str) -> str:
    cleaned = re.sub(r'\[\s*S\d+(?:\s*,\s*S\d+)*\s*\]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# ---------------------- Streamlit Config & CSS ----------------------
st.set_page_config(page_title="RAG — Natural Farming", layout="wide")
st.title("🌾 Prakriti AI")

# Aggressive CSS that targets Streamlit's button DOM and tightens spacing
st.markdown("""
<style>
/* container for the suggested questions block (pull it up a little) */
div.suggest-section {
    text-align: left;
    margin-top: -8px !important;
    margin-bottom: 0px !important;
    padding: 0px !important;
}

/* wrapper we add around each button (remove extra spacing Streamlit adds) */
div.suggest-btn {
    margin: 0px !important;
    padding: 0px !important;
}

/* target the Streamlit button container inside our wrapper and remove its margins */
div.suggest-btn > div.stButton {
    margin: 0px !important;
    padding: 0px !important;
}

/* finally style the actual <button> element inside the .stButton */
div.suggest-btn > div.stButton > button {
    background-color: #f5f7fb !important;
    color: #1b1f23 !important;
    border-radius: 10px !important;
    border: 1px solid #e1e5ea !important;
    padding: 0.45rem 0.9rem !important;    /* tighter internal padding */
    margin: 0.06rem 0 0.06rem 0 !important; /* very small vertical gap */
    width: 60% !important;                /* keep left-aligned and not full width */
    text-align: left !important;
    font-weight: 500 !important;
    white-space: nowrap !important;       /* do not wrap the text */
    line-height: 1.05 !important;
}

/* hover style */
div.suggest-btn > div.stButton > button:hover {
    background-color: #e9eef8 !important;
    border-color: #d0d4da !important;
}

/* remove extra spacing between the title and first button */
div.suggest-section + div {
    margin-top: 0px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- App variables ----------------------
VECTOR_DIR = "vectorstore"
DATA_DIR = "data"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# initialize session state keys used for UI flow
if "history" not in st.session_state:
    st.session_state.history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
# temporary key that holds a suggestion to apply before widget creation
# do not set query_input after widget instantiation
if "apply_suggestion" not in st.session_state:
    st.session_state.apply_suggestion = None
if "trigger_ask" not in st.session_state:
    st.session_state.trigger_ask = False

# ---------------------- Sidebar ----------------------
st.sidebar.header("Ingest Documents / URLs")
uploaded = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
url_input = st.sidebar.text_input("Add URL to ingest")
ingest_btn = st.sidebar.button("Ingest")
clear_index_btn = st.sidebar.button("Clear vectorstore")

if clear_index_btn:
    # remove vectorstore files
    try:
        for fname in ["faiss.index", "metadatas.pkl", "embeddings.npy"]:
            path = os.path.join(VECTOR_DIR, fname)
            if os.path.exists(path):
                os.remove(path)
        st.sidebar.success("Vectorstore cleared.")
    except Exception as e:
        st.sidebar.error(f"Failed to clear vectorstore: {e}")

if ingest_btn:
    ing = Ingestor(vector_dir=VECTOR_DIR)
    messages = []
    if uploaded:
        for f in uploaded:
            save_path = os.path.join(DATA_DIR, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            ing.ingest_pdf(save_path, source_name=f.name)
            messages.append(f"Ingested {f.name}")
    if url_input:
        try:
            ing.ingest_url(url_input)
            messages.append(f"Ingested URL: {url_input}")
        except Exception as e:
            st.sidebar.error(str(e))
    if messages:
        for m in messages:
            st.sidebar.success(m)
    else:
        st.sidebar.info("No files or URLs provided.")
    st.rerun()

# ---------------------- Suggestion handling (SAFE) ----------------------
# If a suggestion was clicked in the previous run, apply it now BEFORE creating the text_input
if st.session_state.get("apply_suggestion"):
    # move suggested text into the widget-backed key before the widget is created
    st.session_state["query_input"] = st.session_state.pop("apply_suggestion")
    # trigger the ask on this run
    st.session_state["trigger_ask"] = True

# ---------------------- Query area ----------------------
st.subheader("Ask about Natural Farming")

# create the text input (widget) AFTER possible pre-fill
query = st.text_input("Your question", key="query_input")

# Suggested questions (stacked vertically)
st.markdown("#### 💬 Suggested questions")
suggested_questions = [
    "What is natural farming?",
    "How does natural farming differ from organic farming?",
    "What are the benefits of natural farming?"
]

st.markdown('<div class="suggest-section">', unsafe_allow_html=True)
for i, sq in enumerate(suggested_questions):
    st.markdown('<div class="suggest-btn">', unsafe_allow_html=True)
    # When clicked, set a temporary key 'apply_suggestion' and rerun.
    # We DO NOT assign directly to st.session_state['query_input'] here (that causes Streamlit error).
    if st.button(sq, key=f"suggested_{i}"):
        st.session_state["apply_suggestion"] = sq
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Other controls
top_k = st.sidebar.slider("Top K retrieval", min_value=1, max_value=8, value=5)
show_sources = st.checkbox("Show retrieved snippets with similarity (developer view)", value=False)

# Ask button (also serves as manual trigger)
if st.button("Ask"):
    st.session_state["trigger_ask"] = True

# ---------------------- Query execution ----------------------
if st.session_state.get("trigger_ask") and st.session_state.get("query_input", "").strip():
    # consume the trigger
    st.session_state["trigger_ask"] = False
    query_text = st.session_state.get("query_input", "").strip()

    idx_path = os.path.join(VECTOR_DIR, "faiss.index")
    meta_path = os.path.join(VECTOR_DIR, "metadatas.pkl")
    emb_path = os.path.join(VECTOR_DIR, "embeddings.npy")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path) and os.path.exists(emb_path)):
        st.error("No vectorstore found. Ingest PDFs or URLs first from the sidebar.")
    else:
        # Load vectorstore
        index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            metadatas = pickle.load(f)
        embeddings = np.load(emb_path)  # shape (N, d)

        # Embed the query
        embedder = SentenceTransformer(EMB_MODEL_NAME)
        q_emb = embedder.encode([query_text]).astype("float32")[0]

        # Search
        D, I = index.search(np.array([q_emb]).astype("float32"), top_k)

        # Build contexts with cosine similarity and prepare prompt contexts
        contexts_for_prompt = []
        contexts_for_ui = []
        for rank, idx in enumerate(I[0], start=1):
            md = metadatas[idx]
            chunk_text = md.get("text", "(chunk text missing)")
            chunk_emb = embeddings[idx].astype("float32")
            sim = cosine_sim(q_emb.astype("float32"), chunk_emb)
            contexts_for_prompt.append({"metadata": md, "text": chunk_text})
            contexts_for_ui.append({
                "rank": rank,
                "source": md.get("source", "unknown"),
                "page": md.get("page", "?"),
                "chunk_index": md.get("chunk_index"),
                "text": chunk_text,
                "cosine": round(sim, 4),
                "faiss_distance": float(D[0][rank-1])
            })

        # Ask Gemini to answer using the contexts (it may include [S#] tokens internally)
        with st.spinner("💬 Generating answer from Gemini..."):
            raw_answer = generate_from_context(query_text, contexts_for_prompt, temperature=0.0)

        # prepare cleaned answer (remove S citations) for display
        cleaned_answer = strip_s_citations(raw_answer)

        # Build unique full-source list (preserve order of appearance)
        seen = set()
        unique_sources = []
        for c in contexts_for_ui:
            src_name = c.get("source", "unknown")
            page = c.get("page", "?")
            key = f"{src_name}||{page}"
            if key not in seen:
                seen.add(key)
                unique_sources.append({"source": src_name, "page": page, "cosine": c.get("cosine")})

        # Save into history (store cleaned answer and full source list)
        st.session_state.history.append({
            "query": query_text,
            "raw_answer": raw_answer,
            "answer": cleaned_answer,
            "sources": unique_sources,
            "contexts_debug": contexts_for_ui
        })

        # Rerun to show the new entry
        st.rerun()

# ---------------------- Render chat history (latest first) ----------------------
for turn in reversed(st.session_state.history):
    st.markdown(f"**Q:** {turn['query']}")
    st.markdown(f"**A:** {turn['answer']}")  # cleaned answer (no [S1] tokens)

    # Separate, clear "Sources used" section with full file names
    st.markdown("**Sources used (full filenames):**")
    for s in turn["sources"]:
        src = s.get("source", "Unknown source")
        page = s.get("page", "?")
        cosine = s.get("cosine", None)
        line = f"- *{src}* (page {page})"
        if cosine is not None:
            line += f" — similarity: {cosine}"
        st.markdown(line)

    # Optional developer view: show retrieved snippet preview and faiss distance if checked
    if show_sources:
        st.markdown("**Developer view — Retrieved snippets (top-k):**")
        for c in turn["contexts_debug"]:
            st.markdown(f"**[Rank {c['rank']}]** — {c['source']} (page {c['page']}) — cosine: {c['cosine']}")
            with st.expander("Snippet preview", expanded=False):
                st.write(c["text"])
                st.caption(f"FAISS L2 distance: {round(c['faiss_distance'], 6)}")

    st.write("---")
