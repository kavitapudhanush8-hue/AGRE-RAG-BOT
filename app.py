"""
FarmAI — Agriculture RAG Chatbot
Multilingual support: English, Hindi, Telugu, Tenglish

Run with:  python -m uvicorn app:app --reload --port 5000
Then open http://localhost:5000 for the chat UI, or POST to /chat directly.
"""

# Load .env BEFORE anything else reads environment variables.
import os
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv not installed; env vars must be set externally

import logging
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag_engine import RagEngine

# ── Logging setup ────────────────────────────────────────────────────────────
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("farmai.app")

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="FarmAI - Agriculture RAG Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RagEngine()

# In-memory conversation history: {conversation_id: [ {role, content, sources, ts}, ... ]}
CONVERSATIONS: dict[str, list[dict]] = {}


# ── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    top_k: int = 4
    language: str | None = None  # "auto" | "en" | "hi" | "te" | "tenglish" | None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[str]
    detected_language: str = "ENGLISH"
    detected_crop: str | None = None
    detected_domain: str | None = None
    answer_source: str = "document"  # "document" | "general_knowledge" | "non_agriculture"
    debug: dict | None = None


# ── Chat endpoint ────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        return ChatResponse(
            conversation_id=req.conversation_id or str(uuid.uuid4()),
            answer="Please enter a question.",
            sources=[],
            detected_language="ENGLISH",
            answer_source="non_agriculture",
        )

    conversation_id = req.conversation_id or str(uuid.uuid4())
    CONVERSATIONS.setdefault(conversation_id, [])

    # Get conversation history for context
    history = CONVERSATIONS[conversation_id]

    try:
        result = engine.answer(
            req.question,
            top_k=req.top_k,
            conversation_history=history,
        )
    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        return ChatResponse(
            conversation_id=conversation_id,
            answer="Sorry, I couldn't process your question right now. Please try again.",
            sources=[],
            detected_language="ENGLISH",
            answer_source="non_agriculture",
        )

    # Store in conversation history
    CONVERSATIONS[conversation_id].append(
        {
            "role": "user",
            "content": req.question,
            "ts": time.time(),
        }
    )
    CONVERSATIONS[conversation_id].append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "ts": time.time(),
        }
    )

    # Limit history length to prevent memory bloat (configurable)
    max_msgs = int(os.environ.get("MAX_CONVERSATION_MESSAGES", "10"))
    max_stored = max_msgs * 2  # store a bit more than we send to LLM
    if len(CONVERSATIONS[conversation_id]) > max_stored:
        CONVERSATIONS[conversation_id] = CONVERSATIONS[conversation_id][-max_msgs:]

    return ChatResponse(
        conversation_id=conversation_id,
        answer=result["answer"],
        sources=result["sources"],
        detected_language=result.get("detected_language", "ENGLISH"),
        detected_crop=result.get("detected_crop"),
        detected_domain=result.get("detected_domain"),
        answer_source=result.get("answer_source", "document"),
        debug=result.get("debug"),
    )


# ── History endpoint ─────────────────────────────────────────────────────────

@app.get("/history/{conversation_id}")
def history(conversation_id: str):
    return {
        "conversation_id": conversation_id,
        "messages": CONVERSATIONS.get(conversation_id, []),
    }


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_indexed": len(engine.chunks),
        "supported_languages": ["English", "Hindi", "Telugu", "Tenglish"],
    }


# ── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FarmAI - Multilingual Agriculture RAG Chatbot</title>
<meta name="description" content="FarmAI: Intelligent multilingual agriculture chatbot supporting English, Hindi, Telugu, and Tenglish. Ask farming questions and get AI-powered answers grounded in agricultural knowledge.">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

  :root {
    --bg-gradient-start: #0f1a0f;
    --bg-gradient-end: #1a2e1a;
    --surface: rgba(255, 255, 255, 0.05);
    --surface-hover: rgba(255, 255, 255, 0.08);
    --glass: rgba(255, 255, 255, 0.06);
    --glass-border: rgba(255, 255, 255, 0.1);
    --primary: #4ade80;
    --primary-glow: rgba(74, 222, 128, 0.15);
    --primary-dark: #22c55e;
    --accent: #a78bfa;
    --accent-glow: rgba(167, 139, 250, 0.1);
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --user-bubble: linear-gradient(135deg, #22c55e, #16a34a);
    --bot-bubble: rgba(255, 255, 255, 0.07);
    --danger: #f87171;
    --warning: #fbbf24;
    --info: #60a5fa;
    --shadow-xl: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Outfit', system-ui, -apple-system, sans-serif;
    background: linear-gradient(160deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 50%, #0d190d 100%);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: row;
    overflow: hidden;
    position: relative;
  }

  /* Animated background particles */
  body::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
      radial-gradient(circle at 20% 80%, var(--primary-glow) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, var(--accent-glow) 0%, transparent 50%),
      radial-gradient(circle at 50% 50%, rgba(74, 222, 128, 0.03) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: bgShift 20s ease-in-out infinite alternate;
  }

  @keyframes bgShift {
    0% { opacity: 0.7; }
    50% { opacity: 1; }
    100% { opacity: 0.7; }
  }

  /* Sidebar */
  .sidebar {
    width: 260px;
    background: rgba(0, 0, 0, 0.2);
    border-right: 1px solid var(--glass-border);
    backdrop-filter: blur(20px);
    display: flex;
    flex-direction: column;
    padding: 16px;
    z-index: 10;
  }

  .sidebar-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--glass-border);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .history-list {
    list-style: none;
    overflow-y: auto;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .history-list::-webkit-scrollbar { width: 4px; }
  .history-list::-webkit-scrollbar-track { background: transparent; }
  .history-list::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 4px; }

  .history-item {
    font-size: 0.85rem;
    padding: 10px 12px;
    background: var(--surface);
    border-radius: 8px;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border: 1px solid transparent;
    transition: all 0.2s;
    color: var(--text);
  }

  .history-item:hover {
    background: var(--surface-hover);
    border-color: var(--primary);
  }

  .main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
  }

  /* Header */
  .header {
    position: relative;
    z-index: 10;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--glass-border);
    background: var(--glass);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .logo-icon {
    font-size: 28px;
    filter: drop-shadow(0 0 8px var(--primary-glow));
    animation: gentlePulse 3s ease-in-out infinite;
  }

  @keyframes gentlePulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
  }

  .header h1 {
    font-size: 1.35rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
  }

  .header-subtitle {
    font-size: 0.7rem;
    color: var(--text-dim);
    font-weight: 400;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  /* Language selector */
  .lang-select-wrapper {
    position: relative;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .lang-label {
    font-size: 0.7rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
  }

  #langSelect {
    appearance: none;
    -webkit-appearance: none;
    background: var(--surface);
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    color: var(--text);
    padding: 6px 30px 6px 10px;
    font-family: inherit;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%2394a3b8'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 10px center;
  }

  #langSelect:hover, #langSelect:focus {
    border-color: var(--primary);
    outline: none;
    box-shadow: 0 0 0 2px var(--primary-glow);
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--primary);
    box-shadow: 0 0 6px var(--primary);
    animation: statusPulse 2s infinite;
  }

  @keyframes statusPulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px var(--primary); }
    50% { opacity: 0.5; box-shadow: 0 0 12px var(--primary); }
  }

  /* Chat area */
  .chat-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    max-width: 860px;
    width: 100%;
    margin: 0 auto;
    padding: 0 16px;
    position: relative;
    z-index: 5;
    overflow: hidden;
  }

  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 20px 4px;
    scroll-behavior: smooth;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  #chat::-webkit-scrollbar { width: 5px; }
  #chat::-webkit-scrollbar-track { background: transparent; }
  #chat::-webkit-scrollbar-thumb {
    background: rgba(74, 222, 128, 0.2);
    border-radius: 10px;
  }
  #chat::-webkit-scrollbar-thumb:hover {
    background: rgba(74, 222, 128, 0.4);
  }

  /* Messages */
  .msg {
    display: flex;
    flex-direction: column;
    max-width: 80%;
    opacity: 0;
    animation: msgIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }

  @keyframes msgIn {
    from { opacity: 0; transform: translateY(12px) scale(0.97); }
    to { opacity: 1; transform: translateY(0) scale(1); }
  }

  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.bot { align-self: flex-start; align-items: flex-start; }

  .msg-label {
    font-size: 0.65rem;
    color: var(--text-dim);
    margin-bottom: 4px;
    padding: 0 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
  }

  .bubble {
    padding: 12px 18px;
    border-radius: 18px;
    line-height: 1.6;
    font-size: 0.92rem;
    position: relative;
    transition: transform 0.15s ease;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .bubble:hover { transform: scale(1.01); }

  .user .bubble {
    background: var(--user-bubble);
    color: #fff;
    border-bottom-right-radius: 6px;
    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.2);
  }

  .bot .bubble {
    background: var(--bot-bubble);
    border: 1px solid var(--glass-border);
    border-bottom-left-radius: 6px;
    backdrop-filter: blur(8px);
  }

  /* Answer source badge */
  .answer-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
    padding: 0 12px;
    flex-wrap: wrap;
  }

  .source-badge {
    font-size: 0.6rem;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .badge-document {
    background: rgba(74, 222, 128, 0.12);
    color: var(--primary);
    border: 1px solid rgba(74, 222, 128, 0.2);
  }

  .badge-general {
    background: rgba(251, 191, 36, 0.12);
    color: var(--warning);
    border: 1px solid rgba(251, 191, 36, 0.2);
  }

  .badge-lang {
    background: rgba(167, 139, 250, 0.12);
    color: var(--accent);
    border: 1px solid rgba(167, 139, 250, 0.2);
  }

  .sources {
    font-size: 0.7rem;
    color: var(--text-dim);
    padding: 0 12px;
    margin-top: 2px;
  }

  .sources a {
    color: var(--primary);
    text-decoration: none;
  }

  /* Thinking animation */
  .thinking-dots { display: flex; gap: 5px; padding: 4px 0; }
  .thinking-dots .dot {
    width: 8px; height: 8px;
    background: var(--primary);
    border-radius: 50%;
    animation: dotBounce 1.4s ease-in-out infinite both;
  }
  .thinking-dots .dot:nth-child(1) { animation-delay: -0.32s; }
  .thinking-dots .dot:nth-child(2) { animation-delay: -0.16s; }
  @keyframes dotBounce {
    0%, 80%, 100% { transform: scale(0.4); opacity: 0.3; }
    40% { transform: scale(1); opacity: 1; }
  }

  /* Input area */
  .input-area {
    padding: 12px 0 20px;
    position: relative;
    z-index: 10;
  }

  .input-container {
    display: flex;
    gap: 10px;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 6px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: border-color 0.3s, box-shadow 0.3s;
  }

  .input-container:focus-within {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px var(--primary-glow), var(--shadow-xl);
  }

  #question {
    flex: 1;
    padding: 14px 18px;
    border: none;
    background: transparent;
    color: var(--text);
    font-size: 0.95rem;
    font-family: inherit;
    outline: none;
    min-width: 0;
  }

  #question::placeholder {
    color: var(--text-dim);
  }

  #sendBtn {
    padding: 12px 24px;
    border: none;
    border-radius: 12px;
    background: var(--user-bubble);
    color: white;
    font-weight: 600;
    font-size: 0.9rem;
    font-family: inherit;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  #sendBtn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(34, 197, 94, 0.3);
  }

  #sendBtn:active {
    transform: translateY(0);
  }

  #sendBtn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }

  /* Welcome message */
  .welcome-card {
    text-align: center;
    padding: 40px 20px;
    animation: fadeInUp 0.6s ease;
  }

  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .welcome-icon { font-size: 48px; margin-bottom: 16px; }

  .welcome-title {
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 8px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .welcome-text {
    color: var(--text-muted);
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 500px;
    margin: 0 auto 20px;
  }

  .lang-chips {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
  }

  .lang-chip {
    font-size: 0.75rem;
    padding: 6px 14px;
    border-radius: 20px;
    background: var(--surface);
    border: 1px solid var(--glass-border);
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s;
  }

  .lang-chip:hover {
    background: var(--primary-glow);
    border-color: var(--primary);
    color: var(--primary);
  }

  /* Responsive */
  @media (max-width: 640px) {
    .sidebar { display: none; }
    .header { padding: 12px 16px; }
    .header h1 { font-size: 1.1rem; }
    .header-subtitle { display: none; }
    .chat-wrapper { padding: 0 8px; }
    .msg { max-width: 90%; }
    .bubble { font-size: 0.88rem; padding: 10px 14px; }
    #sendBtn { padding: 12px 16px; }
    .welcome-card { padding: 30px 16px; }
    .lang-label { display: none; }
  }
</style>
</head>
<body>
  <div class="sidebar">
    <div class="sidebar-title">History</div>
    <ul class="history-list" id="history-list"></ul>
  </div>
  <div class="main-content">
  <div class="header">
    <div class="header-left">
      <span class="logo-icon">&#127806;</span>
      <div>
        <h1>FarmAI Assistant</h1>
        <div class="header-subtitle">Multilingual Agriculture Intelligence</div>
      </div>
    </div>
    <div class="header-right">
      <div class="lang-select-wrapper">
        <span class="lang-label">Language</span>
        <select id="langSelect">
          <option value="auto" selected>&#127760; Auto Detect</option>
          <option value="en">English</option>
          <option value="hi">हिन्दी (Hindi)</option>
          <option value="te">తెలుగు (Telugu)</option>
          <option value="tenglish">Tenglish</option>
        </select>
      </div>
      <div class="status-dot" title="Online"></div>
    </div>
  </div>

  <div class="chat-wrapper">
    <div id="chat">
      <div class="welcome-card">
        <div class="welcome-icon">&#127806;</div>
        <div class="welcome-title">Welcome to FarmAI</div>
        <div class="welcome-text">
          Your multi-crop agriculture assistant. Ask about Chilli, Cotton, Groundnut, Rice,
          and more — soil, irrigation, pests, diseases, fertilizers — in any language.
        </div>
        <div class="lang-chips">
          <span class="lang-chip" onclick="setExample('What is the best soil pH for chilli?')">Chilli</span>
          <span class="lang-chip" onclick="setExample('Cotton crop lo pest control ela cheyyali?')">Cotton</span>
          <span class="lang-chip" onclick="setExample('Groundnut ki fertilizer eppudu ivvali?')">Groundnut</span>
          <span class="lang-chip" onclick="setExample('Rice crop ki irrigation ela manage cheyyali?')">Rice</span>
          <span class="lang-chip" onclick="setExample('What is drip irrigation?')">Irrigation</span>
          <span class="lang-chip" onclick="setExample('What is crop rotation?')">General</span>
        </div>
      </div>
    </div>

    <div class="input-area">
      <form id="form">
        <div class="input-container">
          <input id="question" placeholder="Ask your farming question in any language..." autocomplete="off" />
          <button type="submit" id="sendBtn">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
            Ask
          </button>
        </div>
      </form>
    </div>
  </div>
  </div>

<script>
let conversationId = null;
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const questionInput = document.getElementById('question');
const sendBtn = document.getElementById('sendBtn');
const langSelect = document.getElementById('langSelect');

function setExample(text) {
  questionInput.value = text;
  questionInput.focus();
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  // Clear welcome card on first message
  const welcome = document.querySelector('.welcome-card');
  if (welcome) welcome.remove();

  function addHistoryItem(text) {
    const list = document.getElementById('history-list');
    const li = document.createElement('li');
    li.className = 'history-item';
    li.textContent = text;
    li.title = text;
    // Optional: click history item to ask again
    li.onclick = () => setExample(text);
    list.insertBefore(li, list.firstChild);
  }

  addMessage('user', question);
  addHistoryItem(question);
  questionInput.value = '';
  sendBtn.disabled = true;

  const thinkingId = 'think-' + Date.now();
  addThinking(thinkingId);

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        question,
        conversation_id: conversationId,
        language: langSelect.value
      })
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();
    conversationId = data.conversation_id;
    removeThinking(thinkingId);
    addMessage('bot', data.answer, data.sources, data.detected_language, data.answer_source, data.detected_crop, data.detected_domain);
  } catch (err) {
    removeThinking(thinkingId);
    addMessage('bot', 'Sorry, I could not reach the server. Please try again.', [], 'ENGLISH', 'error');
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
  }
});

function addMessage(role, text, sources, detectedLang, answerSource, detectedCrop, detectedDomain) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;

  // Role label
  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? 'You' : 'FarmAI';
  div.appendChild(label);

  // Bubble
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  div.appendChild(bubble);

  // Meta info for bot messages
  if (role === 'bot' && answerSource && answerSource !== 'error') {
    const meta = document.createElement('div');
    meta.className = 'answer-meta';

    // Language badge
    if (detectedLang) {
      const langBadge = document.createElement('span');
      langBadge.className = 'source-badge badge-lang';
      langBadge.textContent = detectedLang;
      meta.appendChild(langBadge);
    }

    // Crop badge
    if (detectedCrop) {
      const cropBadge = document.createElement('span');
      cropBadge.className = 'source-badge badge-document';
      cropBadge.textContent = '🌱 ' + detectedCrop.charAt(0).toUpperCase() + detectedCrop.slice(1);
      meta.appendChild(cropBadge);
    }

    // Answer source badge
    const srcBadge = document.createElement('span');
    srcBadge.className = 'source-badge';
    if (answerSource === 'document') {
      srcBadge.className += ' badge-document';
      srcBadge.textContent = '📄 Document';
    } else if (answerSource === 'general_knowledge') {
      srcBadge.className += ' badge-general';
      srcBadge.textContent = '💡 General Knowledge';
    } else if (answerSource === 'non_agriculture') {
      srcBadge.className += ' badge-general';
      srcBadge.textContent = '🌾 Agriculture Only';
    }
    meta.appendChild(srcBadge);
    div.appendChild(meta);
  }

  // Sources
  if (sources && sources.length) {
    const s = document.createElement('div');
    s.className = 'sources';
    s.textContent = '📚 Sources: ' + sources.join(', ');
    div.appendChild(s);
  }

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function addThinking(id) {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.id = id;

  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = 'FarmAI';
  div.appendChild(label);

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<div class="thinking-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  div.appendChild(bubble);

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// Handle Enter key
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.dispatchEvent(new Event('submit'));
  }
});
</script>
</body>
</html>
"""
