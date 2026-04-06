//! Web chat UI served at /chat — modern conversational interface.
//!
//! Sends messages to /v1/chat/completions with SSE streaming,
//! renders responses token-by-token with typing animation.

use axum::response::Html;

pub async fn serve_chat() -> Html<&'static str> {
    Html(CHAT_HTML)
}

const CHAT_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLX-Flash Chat</title>
<style>
  :root {
    --bg: #0a0e14; --surface: #131920; --card: #1a2029; --border: #262d38;
    --text: #d4dce8; --dim: #5c6a7a; --accent: #4da6ff; --accent2: #7b61ff;
    --green: #2dd4a8; --user-bg: #1e3a5f; --ai-bg: #1a2029;
    --input-bg: #131920; --hover: #222c38;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; }
  body { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, system-ui, sans-serif; background: var(--bg); color: var(--text); display: flex; flex-direction: column; }

  .topbar { padding: 14px 24px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
  .topbar h1 { font-size: 1.1rem; font-weight: 600; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .topbar .model-name { font-size: 0.8rem; color: var(--dim); padding: 3px 10px; background: var(--card); border-radius: 6px; border: 1px solid var(--border); }
  .topbar .nav { margin-left: auto; display: flex; gap: 8px; }
  .topbar .nav a { color: var(--dim); text-decoration: none; font-size: 0.8rem; padding: 4px 10px; border-radius: 6px; transition: all 0.15s; }
  .topbar .nav a:hover { color: var(--text); background: var(--hover); }

  .messages { flex: 1; overflow-y: auto; padding: 20px 0; scroll-behavior: smooth; }
  .messages::-webkit-scrollbar { width: 6px; }
  .messages::-webkit-scrollbar-track { background: transparent; }
  .messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .msg { max-width: 800px; margin: 0 auto; padding: 0 24px; }
  .msg-row { display: flex; gap: 14px; padding: 16px 0; }
  .msg-row + .msg-row { border-top: 1px solid rgba(38,45,56,0.5); }
  .avatar { width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0; font-weight: 600; }
  .avatar-user { background: var(--user-bg); color: var(--accent); }
  .avatar-ai { background: linear-gradient(135deg, var(--accent), var(--accent2)); color: white; }
  .msg-content { flex: 1; line-height: 1.65; font-size: 0.92rem; min-width: 0; }
  .msg-content p { margin-bottom: 8px; }
  .msg-content p:last-child { margin-bottom: 0; }
  .msg-content code { background: var(--surface); padding: 2px 6px; border-radius: 4px; font-size: 0.85em; font-family: 'SF Mono', Menlo, monospace; }
  .msg-content pre { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; overflow-x: auto; margin: 8px 0; }
  .msg-content pre code { background: none; padding: 0; }
  .typing { display: inline-block; }
  .typing::after { content: ''; display: inline-block; width: 2px; height: 1em; background: var(--accent); margin-left: 1px; animation: blink 0.8s infinite; vertical-align: text-bottom; }
  @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }

  .empty { text-align: center; padding: 80px 24px; }
  .empty h2 { font-size: 1.5rem; font-weight: 600; margin-bottom: 8px; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .empty p { color: var(--dim); font-size: 0.9rem; margin-bottom: 24px; }
  .suggestions { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
  .suggestion { padding: 8px 16px; border-radius: 10px; border: 1px solid var(--border); background: var(--card); color: var(--text); font-size: 0.82rem; cursor: pointer; transition: all 0.15s; }
  .suggestion:hover { border-color: var(--accent); background: var(--hover); }

  .input-area { flex-shrink: 0; border-top: 1px solid var(--border); padding: 16px 24px; background: var(--surface); }
  .input-wrap { max-width: 800px; margin: 0 auto; display: flex; gap: 10px; align-items: flex-end; }
  .input-wrap textarea { flex: 1; background: var(--input-bg); border: 1px solid var(--border); border-radius: 12px; padding: 12px 16px; color: var(--text); font-size: 0.92rem; font-family: inherit; resize: none; outline: none; min-height: 46px; max-height: 200px; line-height: 1.5; transition: border-color 0.15s; }
  .input-wrap textarea:focus { border-color: var(--accent); }
  .input-wrap textarea::placeholder { color: var(--dim); }
  .send-btn { width: 42px; height: 42px; border-radius: 10px; border: none; background: linear-gradient(135deg, var(--accent), var(--accent2)); color: white; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: opacity 0.15s; flex-shrink: 0; }
  .send-btn:hover { opacity: 0.85; }
  .send-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .send-btn svg { width: 18px; height: 18px; }

  .status-bar { max-width: 800px; margin: 0 auto; display: flex; gap: 12px; align-items: center; padding-top: 8px; font-size: 0.7rem; color: var(--dim); }
  .status-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--green); }
</style>
</head>
<body>

<div class="topbar">
  <h1>MLX-Flash</h1>
  <span class="model-name" id="model-badge">loading...</span>
  <div class="nav">
    <a href="/admin">Dashboard</a>
    <a href="/chat">Chat</a>
  </div>
</div>

<div class="messages" id="messages">
  <div class="empty" id="empty">
    <h2>MLX-Flash Chat</h2>
    <p>Running locally on your Mac — private, fast, no cloud</p>
    <div class="suggestions">
      <div class="suggestion" onclick="sendSuggestion(this)">Explain how MoE models work</div>
      <div class="suggestion" onclick="sendSuggestion(this)">Write a Python quicksort</div>
      <div class="suggestion" onclick="sendSuggestion(this)">What makes Apple Silicon good for AI?</div>
    </div>
  </div>
</div>

<div class="input-area">
  <div class="input-wrap">
    <textarea id="input" placeholder="Message MLX-Flash..." rows="1" autofocus></textarea>
    <button class="send-btn" id="send" onclick="sendMessage()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13"/><path d="M22 2L15 22L11 13L2 9L22 2Z"/></svg>
    </button>
  </div>
  <div class="status-bar">
    <div class="status-dot"></div>
    <span id="status-text">Ready</span>
    <span style="margin-left:auto" id="mem-info"></span>
  </div>
</div>

<script>
let messages = [];
let generating = false;

const textarea = document.getElementById('input');
const sendBtn = document.getElementById('send');
const messagesEl = document.getElementById('messages');

// Auto-resize textarea
textarea.addEventListener('input', () => {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
});
textarea.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function sendSuggestion(el) { textarea.value = el.textContent; sendMessage(); }

function addMessage(role, content) {
  document.getElementById('empty')?.remove();
  const row = document.createElement('div');
  row.className = 'msg';
  const isUser = role === 'user';
  row.innerHTML = `<div class="msg-row">
    <div class="avatar ${isUser?'avatar-user':'avatar-ai'}">${isUser?'You':'AI'}</div>
    <div class="msg-content" id="msg-${messages.length}">${formatContent(content)}</div>
  </div>`;
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  messages.push({role, content});
  return document.getElementById('msg-' + (messages.length - 1));
}

function formatContent(text) {
  if (!text) return '<span class="typing"></span>';
  // Basic markdown: code blocks, inline code, paragraphs
  text = text.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
  text = text.replace(/\n\n/g, '</p><p>');
  return '<p>' + text + '</p>';
}

async function sendMessage() {
  const text = textarea.value.trim();
  if (!text || generating) return;

  textarea.value = ''; textarea.style.height = 'auto';
  addMessage('user', text);
  const aiEl = addMessage('assistant', '');
  generating = true;
  sendBtn.disabled = true;
  document.getElementById('status-text').textContent = 'Generating...';

  try {
    const payload = {
      model: 'local',
      messages: messages.map(m => ({role: m.role, content: m.content})),
      stream: true, max_tokens: 1024,
    };

    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const err = await resp.text();
      aiEl.innerHTML = '<p style="color:var(--red)">Error: ' + resp.status + ' — ' + err.substring(0, 200) + '</p>';
      messages[messages.length-1].content = 'Error';
      return;
    }

    // SSE streaming
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') continue;
        try {
          const json = JSON.parse(data);
          const delta = json.choices?.[0]?.delta?.content || '';
          if (delta) {
            fullText += delta;
            aiEl.innerHTML = formatContent(fullText) + '<span class="typing"></span>';
            messagesEl.scrollTop = messagesEl.scrollHeight;
          }
        } catch(e) {}
      }
    }

    // If no SSE, try plain JSON response
    if (!fullText && resp.headers.get('content-type')?.includes('json')) {
      try {
        const json = JSON.parse(buffer || await resp.text());
        fullText = json.choices?.[0]?.message?.content || json.choices?.[0]?.text || '';
      } catch(e) {}
    }

    aiEl.innerHTML = formatContent(fullText || 'No response received.');
    messages[messages.length-1].content = fullText;

  } catch(e) {
    aiEl.innerHTML = '<p style="color:var(--red)">Connection error — is the Python worker running?</p>';
  } finally {
    generating = false;
    sendBtn.disabled = false;
    document.getElementById('status-text').textContent = 'Ready';
  }
}

// Poll model + memory for status bar
async function updateStatus() {
  try {
    const st = await fetch('/status').then(r=>r.json());
    const model = (st.model||'').split('/').pop();
    document.getElementById('model-badge').textContent = model || 'no model';
    const mem = st.memory || {};
    const avail = Math.max((mem.free_gb||0)+(mem.inactive_gb||0)*0.5, 0);
    document.getElementById('mem-info').textContent = avail.toFixed(1)+'GB free / '+((mem.total_gb||0).toFixed(0))+'GB';
  } catch(e) {}
}
updateStatus(); setInterval(updateStatus, 5000);
</script>
</body>
</html>
"##;
