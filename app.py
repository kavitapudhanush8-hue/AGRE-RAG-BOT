"""
Phase 1, 9: Chat API + conversation history.

Run with:  uvicorn app:app --reload --port 8000
Then open http://localhost:8000 for a minimal test chat UI, or POST to
/chat directly.
"""

import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag_engine import RagEngine

app = FastAPI(title="AGRE-RAG-BOT - Agriculture RAG Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RagEngine()

# In-memory conversation history: {conversation_id: [ {role, content, sources, ts}, ... ]}
# Swap this for a real database (e.g. the same Supabase Postgres instance
# used for the vector store) to persist history across server restarts.
CONVERSATIONS = {}


class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    top_k: int = 4


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conversation_id = req.conversation_id or str(uuid.uuid4())
    CONVERSATIONS.setdefault(conversation_id, [])

    result = engine.answer(req.question, top_k=req.top_k)

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

    return ChatResponse(
        conversation_id=conversation_id,
        answer=result["answer"],
        sources=result["sources"],
    )


@app.get("/history/{conversation_id}")
def history(conversation_id: str):
    return {"conversation_id": conversation_id, "messages": CONVERSATIONS.get(conversation_id, [])}


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(engine.chunks)}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AGRE-RAG-BOT - Agriculture Assistant</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&display=swap');
  
  :root {
    --bg-color: #e8f5e9;
    --chat-bg: rgba(255, 255, 255, 0.9);
    --primary: #2e7d32;
    --primary-dark: #1b5e20;
    --bot-bubble: #f1f8e9;
    --text-dark: #1f2d20;
  }
  
  body { 
    font-family: 'Outfit', system-ui, sans-serif; 
    margin: 0; 
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
    color: var(--text-dark);
    perspective: 1000px;
    overflow: hidden;
  }
  
  h1 { 
    color: var(--primary-dark); 
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    animation: float 3s ease-in-out infinite;
    font-size: 2.5rem;
  }
  
  @keyframes float {
    0% { transform: translateY(0px) rotateX(0deg); }
    50% { transform: translateY(-10px) rotateX(5deg); }
    100% { transform: translateY(0px) rotateX(0deg); }
  }
  
  .container {
    width: 90%;
    max-width: 700px;
    transform: rotateX(5deg) rotateY(-2deg) translateZ(20px);
    transition: transform 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
  }
  
  .container:hover {
    transform: rotateX(0deg) rotateY(0deg) translateZ(40px);
  }
  
  #chat { 
    border: 1px solid rgba(255,255,255,0.5);
    border-radius: 16px; 
    padding: 20px; 
    height: 60vh;
    max-height: 500px; 
    overflow-y: auto; 
    background: var(--chat-bg);
    backdrop-filter: blur(10px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.15), inset 0 2px 4px rgba(255,255,255,0.5);
    margin-bottom: 20px;
    scroll-behavior: smooth;
  }
  
  /* Scrollbar styles */
  #chat::-webkit-scrollbar { width: 8px; }
  #chat::-webkit-scrollbar-track { background: rgba(0,0,0,0.05); border-radius: 8px; }
  #chat::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 8px; }
  
  .msg { 
    margin-bottom: 16px; 
    opacity: 0;
    animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
  }
  
  @keyframes popIn {
    0% { opacity: 0; transform: scale(0.7) translateY(30px) rotateX(-20deg); }
    100% { opacity: 1; transform: scale(1) translateY(0) rotateX(0deg); }
  }
  
  .user { text-align: right; transform-origin: right bottom; }
  .bot { text-align: left; transform-origin: left bottom; }
  
  .user .bubble { 
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)); 
    color: white; 
    border-bottom-right-radius: 4px;
    box-shadow: -2px 4px 15px rgba(46, 125, 50, 0.3);
  }
  
  .bot .bubble { 
    background: linear-gradient(135deg, #ffffff, var(--bot-bubble)); 
    color: var(--text-dark); 
    border-bottom-left-radius: 4px;
    box-shadow: 2px 4px 15px rgba(0, 0, 0, 0.08);
  }
  
  .bubble { 
    display: inline-block; 
    padding: 14px 20px; 
    border-radius: 16px; 
    max-width: 80%; 
    white-space: pre-wrap; 
    line-height: 1.5;
    position: relative;
    z-index: 1;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    font-size: 15px;
  }
  
  .bubble:hover {
    transform: scale(1.03) translateZ(10px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
  }
  
  .sources { 
    font-size: 12px; 
    color: #555; 
    margin-top: 8px; 
    opacity: 0.8;
  }
  
  form { 
    display: flex; 
    gap: 12px; 
    background: var(--chat-bg);
    padding: 12px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
  }
  
  input { 
    flex: 1; 
    padding: 16px 20px; 
    border-radius: 12px; 
    border: 2px solid transparent; 
    background: rgba(255,255,255,0.8);
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.08);
    font-size: 16px;
    font-family: inherit;
    transition: all 0.3s ease;
  }
  
  input:focus {
    outline: none;
    border-color: var(--primary);
    background: white;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.05), 0 0 15px rgba(46, 125, 50, 0.3);
    transform: translateZ(5px);
  }
  
  button { 
    padding: 14px 28px; 
    border: none; 
    border-radius: 12px; 
    background: linear-gradient(to bottom, var(--primary), var(--primary-dark)); 
    color: white; 
    font-weight: 600;
    font-size: 16px;
    font-family: inherit;
    cursor: pointer; 
    box-shadow: 0 6px 0 #144017, 0 10px 20px rgba(46, 125, 50, 0.4);
    transition: all 0.1s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
    position: relative;
    top: 0;
  }
  
  button:active {
    top: 6px;
    box-shadow: 0 0px 0 #144017, 0 4px 10px rgba(46, 125, 50, 0.4);
    transform: translateZ(-5px);
  }
  
  /* Bot thinking animation */
  .thinking { display: flex; gap: 6px; padding: 4px; }
  .dot { width: 10px; height: 10px; background: var(--primary); border-radius: 50%; animation: bounce 1.4s infinite ease-in-out both; }
  .dot:nth-child(1) { animation-delay: -0.32s; }
  .dot:nth-child(2) { animation-delay: -0.16s; }
  @keyframes bounce { 
    0%, 80%, 100% { transform: scale(0) translateY(0); } 
    40% { transform: scale(1) translateY(-8px); } 
  }
</style>
</head>
<body>
  <h1>&#127806; AGRE-RAG-BOT</h1>
  <div class="container">
    <div id="chat"></div>
    <form id="form">
      <input id="question" placeholder="Ask AGRE-RAG-BOT your farming question..." autocomplete="off" />
      <button type="submit">Ask</button>
    </form>
  </div>
<script>
let conversationId = null;
const chat = document.getElementById('chat');

// Initial greeting
setTimeout(() => {
  addMessage('bot', 'Hello! \U0001f331 I am AGRE-RAG-BOT \u2014 your smart agriculture assistant! Ask me anything about crops, soil, pests, irrigation, market prices, or government schemes. How can I help you today?');
}, 500);

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('question');
  const question = input.value.trim();
  if (!question) return;
  addMessage('user', question);
  input.value = '';
  
  const thinkingId = 'think-' + Date.now();
  addThinking(thinkingId);
  
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, conversation_id: conversationId})
    });
    const data = await res.json();
    conversationId = data.conversation_id;
    removeThinking(thinkingId);
    addMessage('bot', data.answer, data.sources);
  } catch (err) {
    removeThinking(thinkingId);
    addMessage('bot', 'Error: could not reach the server.');
  }
});

function addMessage(role, text, sources) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  div.appendChild(bubble);
  if (sources && sources.length) {
    const s = document.createElement('div');
    s.className = 'sources';
    s.textContent = 'Sources: ' + sources.join(', ');
    div.appendChild(s);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function addThinking(id) {
  const div = document.createElement('div');
  div.className = 'msg bot';
  div.id = id;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<div class="thinking"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  div.appendChild(bubble);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}
</script>
</body>
</html>
"""
