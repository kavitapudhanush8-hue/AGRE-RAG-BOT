"""
Phase 2-5: Knowledge base -> chunking -> embeddings -> vector store.

This script reads every .txt file in data/, splits it into paragraph-level
chunks, builds a TF-IDF vector for each chunk, and saves everything to
index.pkl so the chat app can load it instantly at request time.

TF-IDF is used here instead of a neural embedding model so the whole
pipeline runs offline with zero extra downloads and zero API keys. It is
a real, working retriever -- just less semantically powerful than a
neural embedding model. See README.md ("Swapping in real embeddings")
for how to upgrade this to OpenAI/Gemini/sentence-transformers embeddings
plus a proper vector database (e.g. Supabase + pgvector) without
changing the rest of the app.
"""

import os
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.pkl")


def load_documents():
    """Read every .txt file in data/. Returns list of (source_name, text)."""
    docs = []
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append((filename, text))
    return docs


def extract_title(text, fallback):
    match = re.search(r"^TITLE:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def chunk_document(text):
    """
    Split a document into chunks along its ALL-CAPS section headers
    (e.g. "IRRIGATION", "COMMON PESTS"). Falls back to paragraph
    splitting if no headers are found. Each chunk keeps its section
    heading so retrieved context stays self-describing.
    """
    # Split right before lines that look like section headers:
    # short, all-caps, no trailing punctuation.
    pattern = re.compile(r"\n(?=[A-Z][A-Z /\(\)]{3,60}\n)")
    parts = pattern.split(text.strip())
    chunks = [p.strip() for p in parts if p.strip()]
    if len(chunks) <= 1:
        # Fallback: split on blank lines
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Drop tiny fragments (e.g. a lone "TITLE: ..." line) that add noise
    # to retrieval without carrying useful content.
    chunks = [c for c in chunks if len(c.split()) > 8]
    return chunks


def build_index():
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No .txt documents found in {DATA_DIR}")

    all_chunks = []       # chunk text
    chunk_sources = []    # (source_filename, doc_title) per chunk

    titles = []  # one title string per chunk, used for the topic vectorizer
    for filename, text in docs:
        title = extract_title(text, fallback=filename)
        for chunk in chunk_document(text):
            all_chunks.append(chunk)
            chunk_sources.append({"file": filename, "title": title})
            titles.append(title)

    # Two separate TF-IDF spaces, combined at query time (see rag_engine.py):
    #  - content_vectorizer: matches the specific section text (what actually
    #    answers "when should I sow" vs "what fertilizer").
    #  - topic_vectorizer: matches the crop/topic name itself, so a query
    #    mentioning "groundnut" is nudged toward groundnut chunks even when
    #    the section wording alone wouldn't stand out.
    content_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    content_matrix = content_vectorizer.fit_transform(all_chunks)

    topic_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    topic_matrix = topic_vectorizer.fit_transform(titles)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump(
            {
                "content_vectorizer": content_vectorizer,
                "content_matrix": content_matrix,
                "topic_vectorizer": topic_vectorizer,
                "topic_matrix": topic_matrix,
                "chunks": all_chunks,
                "chunk_sources": chunk_sources,
            },
            f,
        )

    print(f"Indexed {len(all_chunks)} chunks from {len(docs)} documents.")
    print(f"Saved index to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
