import os
from pathlib import Path
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

load_dotenv()

st.set_page_config(page_title="RAG Mini-Chatbot (Gemini)", page_icon="💬")
st.title("💬 RAG Mini-Chatbot (Gemini)")
st.caption("Answers questions about a small document using retrieval + citations.")

@st.cache_data
def list_model_names(api_key: str) -> list[str]:
    client = genai.Client(api_key=api_key)
    names = []
    for m in client.models.list():
        # m.name looks like "models/gemini-..."
        if getattr(m, "name", None):
            names.append(m.name.replace("models/", ""))
    return sorted(set(names))

# ---- Debug / visibility helpers ----
api_key = os.getenv("GEMINI_API_KEY", "")
with st.sidebar:
    st.subheader("Debug")
    st.write("GEMINI_API_KEY loaded:", "✅" if api_key else "❌")
    st.write("Working directory:", os.getcwd())

    model_name = "gemini-2.5-flash"  # fallback default
    if api_key:
        available = list_model_names(api_key)
        if available:
            model_name = st.selectbox("Model", available, index=min(0, len(available)-1))
        else:
            st.warning("Could not list models; using default.")

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(0, end - overlap)
        if end == len(text):
            break
    return chunks

def basic_injection_check(user_text: str) -> bool:
    red_flags = [
        "ignore previous",
        "system prompt",
        "developer message",
        "reveal your instructions",
        "api key",
        "password",
        "secret",
    ]
    t = user_text.lower()
    return any(flag in t for flag in red_flags)

@st.cache_data
def load_doc(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

@st.cache_resource
def build_index(doc_text: str):
    chunks = chunk_text(doc_text)
    if not chunks:
        raise ValueError("Document produced 0 chunks. Is it empty?")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    return chunks, model, index

def retrieve(chunks, model, index, question: str, k: int):
    k = min(k, len(chunks))
    q = model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q, k)

    results = []
    for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
        results.append({"rank": rank, "score": float(s), "chunk": chunks[int(i)]})
    return results

def make_prompt(question: str, retrieved: list[dict]) -> str:
    citations_block = "\n\n".join([f"[C{r['rank']}]\n{r['chunk']}" for r in retrieved])
    return f"""You are a helpful assistant. Answer the user's question using ONLY the provided citations.
If the answer is not contained in the citations, say: "I don't know based on the provided document."

User question:
{question}

Citations:
{citations_block}

Instructions:
- Use only the citations above.
- When you use information, include citation markers like [C1], [C2].
- Keep the answer concise and factual.
"""

# ---- UI ----
doc_path = st.text_input("Document path", "data/zzz_notes.md")
k = st.slider("Top-k chunks to retrieve", 2, 8, 4, 1)

# Load and build retriever lazily so errors show nicely
chunks = model = index = None

if st.button("Build index"):
    st.sidebar.error("✅ Build index button code is running")
    try:
        doc_text = load_doc(doc_path)
        st.sidebar.write("Doc path (resolved):", str(Path(doc_path).resolve()))
        st.sidebar.write("Doc length (chars):", len(doc_text))
        st.sidebar.write("Doc preview:", repr(doc_text[:200]))
        chunks, model, index = build_index(doc_text)
        st.success(f"Indexed {len(chunks)} chunks.")
    except Exception as e:
        st.exception(e)

question = st.text_input("Ask a question about the document")

if st.button("Ask"):
    try:
        if not question.strip():
            st.warning("Type a question first.")
            st.stop()

        if basic_injection_check(question):
            st.warning("⚠️ Possible prompt-injection/sensitive request detected. Rephrase your question.")
            st.stop()

        if not api_key:
            st.error("Missing GEMINI_API_KEY. Put it in rag_prototype_streamlit/.env and restart Streamlit.")
            st.stop()

        doc_text = load_doc(doc_path)
        chunks, model, index = build_index(doc_text)

        retrieved = retrieve(chunks, model, index, question, k=k)
        prompt = make_prompt(question, retrieved)

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        answer = (resp.text or "").strip()

        st.subheader("Answer")
        st.write(answer if answer else "(No text returned.)")

        st.subheader("Retrieved citations")
        for r in retrieved:
            st.markdown(f"**[C{r['rank']}] score={r['score']:.3f}**")
            st.code(r["chunk"])

    except Exception as e:
        st.exception(e)

        st.info("Tip: Put some text in data/zzz_notes.md (even a few bullets). Then click Build index again.")