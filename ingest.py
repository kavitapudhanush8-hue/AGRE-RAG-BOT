"""
Knowledge base ingestion: documents -> chunks -> TF-IDF index.

Reads every .txt file in data/, splits into paragraph-level chunks, auto-detects
crop and domain metadata from filename + content, builds TF-IDF vectors, and
saves everything to index.pkl.

Adding a new crop only requires dropping a new .txt file in data/ and re-running
this script.  No code changes needed.
"""

import os
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.pkl")


# ── Crop & domain detection from filenames/content ──────────────────────────

# Known crops (extend this list as you add new documents)
KNOWN_CROPS = {
    "chilli", "chili", "cotton", "groundnut", "peanut", "rice", "paddy",
    "tomato", "maize", "corn", "banana", "sugarcane", "turmeric", "mango",
    "wheat", "soybean", "sorghum", "jowar", "bajra", "pearl millet",
    "finger millet", "ragi", "sesame", "mustard", "sunflower", "onion",
    "potato", "brinjal", "eggplant", "okra", "papaya", "guava", "grape",
    "citrus", "orange", "lemon",
}

# Known agriculture domains
KNOWN_DOMAINS = {
    "pest_management", "pest", "pests", "ipm",
    "crop_disease", "crop_diseases", "disease", "diseases",
    "irrigation", "water_management", "water",
    "soil", "soil_management",
    "fertilizer", "fertilizer_management", "nutrient", "nutrition",
    "seed", "seed_selection",
    "sowing", "transplanting", "nursery",
    "weed", "weed_management",
    "harvesting", "post_harvest", "harvest",
    "organic_farming", "organic",
    "general", "cultivation", "farming",
}

# Map variations to canonical names
CROP_NORMALIZE = {
    "chili": "chilli", "peanut": "groundnut", "paddy": "rice",
    "corn": "maize", "jowar": "sorghum", "bajra": "pearl_millet",
    "eggplant": "brinjal",
}

DOMAIN_NORMALIZE = {
    "pest": "pest_management", "pests": "pest_management", "ipm": "pest_management",
    "disease": "crop_diseases", "diseases": "crop_diseases", "crop_disease": "crop_diseases",
    "water": "irrigation", "water_management": "irrigation",
    "soil": "soil_management",
    "fertilizer": "fertilizer_management", "nutrient": "fertilizer_management",
    "nutrition": "fertilizer_management",
    "seed": "seed_selection",
    "weed": "weed_management",
    "harvest": "harvesting", "post_harvest": "post_harvest_management",
    "organic": "organic_farming",
    "cultivation": "general", "farming": "general",
    "nursery": "nursery_management", "transplanting": "transplanting",
}


def detect_crop_from_text(filename: str, text: str) -> str:
    """Auto-detect the primary crop from filename and document content."""
    fname_lower = filename.lower().replace(".txt", "").replace("_", " ")
    fname_domain = filename.lower().replace(".txt", "").replace(" ", "_")
    
    # If the filename is primarily a domain (e.g. irrigation.txt, crop_diseases.txt),
    # it is a cross-crop document, so crop is general.
    for domain in sorted(KNOWN_DOMAINS, key=len, reverse=True):
        if domain in fname_domain:
            return "general"

    # Check filename first (most reliable signal for crop docs)
    for crop in KNOWN_CROPS:
        if crop in fname_lower:
            return CROP_NORMALIZE.get(crop, crop)

    # Check the TITLE line specifically
    match = re.search(r"^TITLE:\s*(.+)$", text, re.MULTILINE)
    if match:
        title_lower = match.group(1).lower()
        for crop in sorted(KNOWN_CROPS, key=len, reverse=True):
            if crop in title_lower:
                return CROP_NORMALIZE.get(crop, crop)

    return "general"


def detect_domain_from_text(filename: str, text: str) -> str:
    """Auto-detect the agriculture domain from filename and content."""
    fname_lower = filename.lower().replace(".txt", "").replace(" ", "_")

    # Check filename for domain keywords
    for domain in sorted(KNOWN_DOMAINS, key=len, reverse=True):
        if domain in fname_lower:
            return DOMAIN_NORMALIZE.get(domain, domain)

    # For crop-specific files, the domain is "cultivation" (general for that crop)
    first_500 = text[:500].lower()
    if "cultivation" in first_500 or "guide" in first_500:
        return "cultivation"

    return "general"


def detect_crop_from_chunk(chunk_text: str, doc_crop: str) -> str:
    """
    Refine crop detection at the chunk level.
    A chunk within a cross-crop document (e.g. pest_management.txt) might
    specifically discuss a particular crop.
    """
    if doc_crop != "general":
        return doc_crop

    # For general/cross-crop docs, check if the chunk mentions a specific crop
    lower = chunk_text.lower()
    mentioned_crops = []
    for crop in KNOWN_CROPS:
        if crop in lower:
            canonical = CROP_NORMALIZE.get(crop, crop)
            if canonical not in mentioned_crops:
                mentioned_crops.append(canonical)

    if len(mentioned_crops) == 1:
        return mentioned_crops[0]

    return doc_crop  # stays "general" if 0 or multiple crops


# ── Document loading and chunking ────────────────────────────────────────────

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
    Split a document into chunks along its ALL-CAPS section headers.
    Falls back to paragraph splitting if no headers are found.
    """
    pattern = re.compile(r"\n(?=[A-Z][A-Z /\(\)]{3,60}\n)")
    parts = pattern.split(text.strip())
    chunks = [p.strip() for p in parts if p.strip()]
    if len(chunks) <= 1:
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = [c for c in chunks if len(c.split()) > 8]
    return chunks


# ── Index building ───────────────────────────────────────────────────────────

def build_index():
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No .txt documents found in {DATA_DIR}")

    all_chunks = []
    chunk_sources = []
    titles = []

    for filename, text in docs:
        title = extract_title(text, fallback=filename)
        doc_crop = detect_crop_from_text(filename, text)
        doc_domain = detect_domain_from_text(filename, text)

        for chunk in chunk_document(text):
            # Refine crop at chunk level for cross-crop documents
            chunk_crop = detect_crop_from_chunk(chunk, doc_crop)

            all_chunks.append(chunk)
            chunk_sources.append({
                "file": filename,
                "title": title,
                "crop": chunk_crop,
                "domain": doc_domain,
            })
            titles.append(title)

        print(f"  {filename:30s} -> crop={doc_crop:12s} domain={doc_domain}")

    # Two TF-IDF spaces combined at query time
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

    # Summary
    crops = set(s["crop"] for s in chunk_sources)
    domains = set(s["domain"] for s in chunk_sources)
    print(f"\nIndexed {len(all_chunks)} chunks from {len(docs)} documents.")
    print(f"Crops detected: {sorted(crops)}")
    print(f"Domains detected: {sorted(domains)}")
    print(f"Saved index to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
