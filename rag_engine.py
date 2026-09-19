"""
Phase 6-8: Retrieval -> LLM generation -> source/citation system.

RagEngine.answer(question) does the full pipeline:
  question -> embed -> similarity search -> top chunks -> LLM -> answer + sources
"""

import os
import pickle
import random
import re

from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.pkl")

BOT_NAME = "AGRE-RAG-BOT"

SYSTEM_PROMPT = """You are AGRE-RAG-BOT, a friendly and knowledgeable agriculture assistant for farmers, students, and agribusiness professionals.

Your goal is to ALWAYS provide a helpful, accurate, and practical answer to every agriculture-related question.

Rules:
1. PRIMARY SOURCE: If a CONTEXT block is provided, use it as your main reference and synthesize a clear, direct answer from it.
2. GENERAL KNOWLEDGE FALLBACK: If the context is absent, thin, or doesn't fully answer the question, draw on your broad agricultural expertise to provide the best answer you can. Cover topics like crop cultivation, soil science, pest/disease management, irrigation, fertilizers, market prices, agri-business, government schemes, weather effects on crops, post-harvest handling, and organic farming.
3. MARKET & PRICE QUESTIONS: For market price or commodity questions, provide general guidance (typical price ranges, factors affecting prices, where to check live prices like eNAM or APMC portals) even if you don't have real-time data. Always clarify that prices vary by region and season.
4. GREETINGS & SMALL TALK: Respond warmly and naturally, then invite the user to ask an agriculture question.
5. Be concise but thorough. Use bullet points or short paragraphs for clarity.
6. Do NOT mention 'the context' or 'the documents' in your answer — answer as a knowledgeable agriculture expert would.
7. NEVER say 'I don't know' or refuse to answer an agriculture question. Always provide the most helpful response possible.
8. End difficult answers with a suggestion to verify with a local agricultural extension officer or official government portal for the most current data.
"""

GREETING_KEYWORDS = {
    "hi", "hello", "hey", "hiya", "howdy", "greetings", "good morning",
    "good afternoon", "good evening", "good night", "sup",
    "namaste", "vanakkam", "namaskar",
}

GREETING_RESPONSES = [
    "Hello! \U0001f331 Welcome to AGRE-RAG-BOT \u2014 your smart agriculture assistant!\n\nI can help you with:\n\u2022 Crop cultivation & sowing advice\n\u2022 Pest & disease management\n\u2022 Soil health & fertilizer recommendations\n\u2022 Irrigation guidance\n\u2022 Market prices & agri-business tips\n\u2022 Government schemes & subsidies\n\nWhat's your farming question today?",
    "Hi there! \U0001f33e I'm AGRE-RAG-BOT, your agriculture expert.\n\nAsk me anything about farming, crops, soil, pests, irrigation, or market prices \u2014 I'm here to help!",
    "Namaste! \U0001f91f Welcome to AGRE-RAG-BOT!\n\nI'm your agriculture assistant. Whether you need help with crop selection, disease control, fertilizers, or market information \u2014 just ask!",
]


class RagEngine:
    def __init__(self, index_path=INDEX_PATH):
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No index found at {index_path}. Run `python ingest.py` first."
            )
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.content_vectorizer = data["content_vectorizer"]
        self.content_matrix = data["content_matrix"]
        self.topic_vectorizer = data["topic_vectorizer"]
        self.topic_matrix = data["topic_matrix"]
        self.chunks = data["chunks"]
        self.chunk_sources = data["chunk_sources"]

    def retrieve(self, question, top_k=4, min_score=0.05, topic_weight=0.35):
        """Step 6: embed the question and run similarity search.

        Combines a content-similarity score (which section matches the
        question) with a smaller topic-similarity score (which crop/topic
        the question is about), so results land on the right crop's
        relevant section rather than either signal alone.
        """
        content_scores = cosine_similarity(
            self.content_vectorizer.transform([question]), self.content_matrix
        ).flatten()
        topic_scores = cosine_similarity(
            self.topic_vectorizer.transform([question]), self.topic_matrix
        ).flatten()
        scores = content_scores + topic_weight * topic_scores
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        seen_titles = set()
        for i in ranked[: top_k * 3]:  # look a bit wider, then dedupe by source
            if scores[i] < min_score:
                continue
            results.append(
                {
                    "chunk": self.chunks[i],
                    "title": self.chunk_sources[i]["title"],
                    "file": self.chunk_sources[i]["file"],
                    "score": float(scores[i]),
                }
            )
            if len(results) >= top_k:
                break
        return results

    def build_prompt(self, question, retrieved):
        context_block = "\n\n---\n\n".join(
            f"[Source: {r['title']}]\n{r['chunk']}" for r in retrieved
        )
        return (
            f"CONTEXT:\n{context_block}\n\n"
            f"FARMER'S QUESTION:\n{question}\n\n"
            f"Answer the farmer's question using only the context above."
        )

    def _is_greeting(self, question: str) -> bool:
        """Return True if the message is a simple greeting or small talk."""
        q = question.lower().strip().rstrip("!?.")
        if q in GREETING_KEYWORDS:
            return True
        words = q.split()
        if words and words[0] in GREETING_KEYWORDS and len(words) <= 6:
            return True
        return False

    def generate_answer(self, question, retrieved):
        """Step 7: call the LLM with question + retrieved context.

        Now always provides an answer:
        - If retrieved chunks exist, builds a rich prompt from them.
        - If no chunks found, sends the question to the LLM with a note
          that no local data matched, so it answers from general ag knowledge.
        """
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            if retrieved:
                bullets = "\n".join(f"- {r['chunk'].splitlines()[0]}" for r in retrieved)
                return (
                    "[No GEMINI_API_KEY set - showing raw retrieved context]\n\n"
                    f"{bullets}\n\n{retrieved[0]['chunk']}"
                )
            return (
                "[No GEMINI_API_KEY set] I don't have local data on that topic. "
                "Please set GEMINI_API_KEY so I can answer from general agriculture knowledge."
            )

        if retrieved:
            prompt = self.build_prompt(question, retrieved)
        else:
            prompt = (
                "CONTEXT:\n[No matching entries found in the local knowledge base for this question.]\n\n"
                f"FARMER'S QUESTION:\n{question}\n\n"
                "Please answer this agriculture question using your general agricultural knowledge. "
                "Be practical, thorough, and helpful. If the question is about market prices, "
                "provide typical ranges, influencing factors, and suggest authoritative portals "
                "(eNAM, APMC, Agmarknet) for live data."
            )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            model_name = os.environ.get("AGRE_MODEL", "gemini-3.6-flash")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.4,
                ),
            )
            return response.text
        except ImportError:
            # Fallback to legacy google-generativeai package
            import google.generativeai as genai_legacy
            import warnings
            warnings.filterwarnings("ignore")
            genai_legacy.configure(api_key=api_key)
            model_name = os.environ.get("AGRE_MODEL", "gemini-3.6-flash")
            model = genai_legacy.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT,
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if retrieved:
                return f"LLM call failed ({e}). Raw context:\n\n{retrieved[0]['chunk']}"
            return f"LLM call failed ({e}). Please try again."

    def answer(self, question, top_k=4):
        """Full pipeline: question -> retrieval -> generation -> sources.

        Handles three cases:
        1. Greeting / small talk  -> warm greeting response, skip RAG.
        2. Question with matching chunks -> RAG-augmented LLM answer.
        3. Question with no matching chunks -> LLM answers from general knowledge.
        """
        # Case 1: greeting
        if self._is_greeting(question):
            return {
                "question": question,
                "answer": random.choice(GREETING_RESPONSES),
                "sources": [],
                "retrieved_chunks": [],
            }

        # Case 2 & 3: agriculture question
        retrieved = self.retrieve(question, top_k=top_k)
        answer_text = self.generate_answer(question, retrieved)
        sources = sorted({r["title"] for r in retrieved})
        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "retrieved_chunks": retrieved,  # useful for debugging/UI transparency
        }


if __name__ == "__main__":
    print(f"\n\U0001f33e Welcome to {BOT_NAME}! Type 'quit' to exit.")
    engine = RagEngine()
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"quit", "exit"}:
            print(f"{BOT_NAME}: Goodbye! Happy farming! \U0001f331")
            break
        result = engine.answer(q)
        print(f"\n{BOT_NAME}: " + result["answer"])
        if result["sources"]:
            print("\nSources:", ", ".join(result["sources"]))
