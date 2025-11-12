import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_API_KEY)
DEFAULT_MODEL = "gemini-2.5-flash"

def generate_from_context(question: str, contexts: list, model: str = DEFAULT_MODEL, temperature: float = 0.0):
    """
    Generates an answer with explicit citations like [S1], [S2].
    contexts: list of {"metadata": {...}, "text": "chunk text"}
    """

    # Build structured prompt for Gemini
    system_prompt = (
        "You are an assistant that answers questions ONLY using the provided contexts. "
        "Each context is labeled with an ID like [S1], [S2]. "
        "When you use information from a context, include the label in your answer "
        "(for example: 'Natural farming improves soil health [S1]'). "
        "If the answer is not in the contexts, say 'I don’t know.' "
        "Keep answers factual, concise, and avoid unrelated details."
    )

    # Combine the contexts into a single text block with numbered labels
    labeled_contexts = []
    for i, c in enumerate(contexts, start=1):
        md = c.get("metadata", {})
        src = md.get("source", "unknown source")
        page = md.get("page", "?")
        text = c.get("text", "")
        labeled_contexts.append(f"[S{i}] Source: {src} (page {page})\n{text}")

    full_context = "\n\n---\n\n".join(labeled_contexts)

    prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXTS:\n{full_context}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"YOUR ANSWER (with [S#] citations):"
    )

    model_client = genai.GenerativeModel(model)
    response = model_client.generate_content(prompt, generation_config={"temperature": temperature})
    return getattr(response, "text", str(response))

