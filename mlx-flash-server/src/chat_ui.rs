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
  .model-select { background: var(--card); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 5px 12px; font-size: 0.8rem; font-family: inherit; cursor: pointer; outline: none; -webkit-appearance: none; max-width: 280px; }
  .model-select:focus { border-color: var(--accent); }
  .model-select option { background: var(--card); color: var(--text); }

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
  .msg-content h1,.msg-content h2,.msg-content h3 { font-weight: 600; margin: 12px 0 6px; }
  .msg-content h1 { font-size: 1.2em; } .msg-content h2 { font-size: 1.1em; } .msg-content h3 { font-size: 1em; color: var(--accent); }
  .msg-content strong { font-weight: 600; color: #fff; }
  .msg-content em { font-style: italic; color: var(--dim); }
  .msg-content ul,.msg-content ol { margin: 6px 0 6px 20px; } .msg-content li { margin: 3px 0; }
  .msg-content table { border-collapse: collapse; margin: 8px 0; font-size: 0.85em; width: 100%; }
  .msg-content th { background: var(--surface); font-weight: 600; text-align: left; }
  .msg-content th,.msg-content td { padding: 6px 12px; border: 1px solid var(--border); }
  .msg-content tr:nth-child(even) { background: rgba(19,25,32,0.5); }
  .msg-content hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
  .msg-content blockquote { border-left: 3px solid var(--accent); padding-left: 12px; color: var(--dim); margin: 8px 0; }
  .typing { display: inline-block; }
  .typing::after { content: ''; display: inline-block; width: 2px; height: 1em; background: var(--accent); margin-left: 1px; animation: blink 0.8s infinite; vertical-align: text-bottom; }
  @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }

  .pressure-banner { max-width: 800px; margin: 0 auto 8px; padding: 8px 16px; border-radius: 8px; font-size: 0.78rem; line-height: 1.4; display: none; }
  .pressure-warning { background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.2); color: var(--yellow); display: block; }
  .pressure-critical { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2); color: var(--red); display: block; }
  .gen-progress { display: none; align-items: center; gap: 8px; font-size: 0.72rem; color: var(--dim); }
  .gen-progress.active { display: flex; }
  .gen-progress .tok-counter { color: var(--accent); font-weight: 600; font-variant-numeric: tabular-nums; }
  .gen-progress .mem-indicator { font-variant-numeric: tabular-nums; }
  .pulse-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: pulse-gen 1s infinite; }
  @keyframes pulse-gen { 0%,100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.8); } }

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
  <select id="model-select" class="model-select" onchange="switchModel(this.value)">
    <option value="">Loading models...</option>
  </select>
  <div class="nav">
    <a href="/admin">Dashboard</a>
    <a href="/chat">Chat</a>
  </div>
</div>

<div id="pressure-banner" class="pressure-banner"></div>

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
    <div class="status-dot" id="status-dot"></div>
    <span id="status-text">Ready</span>
    <div class="gen-progress" id="gen-progress">
      <div class="pulse-dot"></div>
      <span class="tok-counter" id="gen-tokens">0 tok</span>
      <span class="mem-indicator" id="gen-mem"></span>
    </div>
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
  // Markdown renderer: code blocks, tables, lists, bold, italic, headers, hr, blockquotes
  let html = text;
  // Code blocks (must be first — protect from other transforms)
  const codeBlocks = [];
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push('<pre><code>' + code.replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</code></pre>');
    return '\x00CB' + idx + '\x00';
  });
  // Tables (pipe-delimited)
  html = html.replace(/(?:^|\n)(\|.+\|)\n(\|[-| :]+\|)\n((?:\|.+\|\n?)+)/gm, (_, header, sep, body) => {
    const ths = header.split('|').filter(c=>c.trim()).map(c=>'<th>'+c.trim()+'</th>').join('');
    const rows = body.trim().split('\n').map(r => {
      const tds = r.split('|').filter(c=>c.trim()).map(c=>'<td>'+c.trim()+'</td>').join('');
      return '<tr>'+tds+'</tr>';
    }).join('');
    return '<table><thead><tr>'+ths+'</tr></thead><tbody>'+rows+'</tbody></table>';
  });
  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // Horizontal rules
  html = html.replace(/^---+$/gm, '<hr>');
  // Blockquotes
  html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
  // Unordered lists
  html = html.replace(/(?:^|\n)((?:- .+\n?)+)/gm, (_, block) => {
    const items = block.trim().split('\n').map(l => '<li>'+l.replace(/^- /,'')+'</li>').join('');
    return '<ul>'+items+'</ul>';
  });
  // Ordered lists
  html = html.replace(/(?:^|\n)((?:\d+\. .+\n?)+)/gm, (_, block) => {
    const items = block.trim().split('\n').map(l => '<li>'+l.replace(/^\d+\. /,'')+'</li>').join('');
    return '<ol>'+items+'</ol>';
  });
  // Bold + italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Paragraphs (double newline)
  html = html.replace(/\n\n+/g, '</p><p>');
  // Single newlines to <br> (except inside block elements)
  html = html.replace(/\n/g, '<br>');
  // Restore code blocks
  html = html.replace(/\x00CB(\d+)\x00/g, (_, idx) => codeBlocks[parseInt(idx)]);
  return '<p>' + html + '</p>';
}

let genPollInterval = null;
let tokBefore = 0;

function startGenProgress() {
  const el = document.getElementById('gen-progress');
  el.classList.add('active');
  document.getElementById('status-dot').style.background = 'var(--accent)';
  tokBefore = 0;
  // Snapshot current tokens
  fetch('/status').then(r=>r.json()).then(st => { tokBefore = (st.stats||{}).tokens_generated||0; }).catch(()=>{});
  // Poll every 500ms during generation
  genPollInterval = setInterval(async () => {
    try {
      const st = await fetch('/status').then(r=>r.json());
      const tokNow = (st.stats||{}).tokens_generated||0;
      const genTok = tokNow - tokBefore;
      document.getElementById('gen-tokens').textContent = genTok + ' tok';
      const mem = st.memory||{};
      const avail = ((mem.free_gb||0)+(mem.inactive_gb||0)*0.5).toFixed(1);
      const pressure = (mem.pressure||'Normal').toString();
      document.getElementById('gen-mem').textContent = avail+'GB free';
      document.getElementById('gen-mem').style.color = pressure==='Critical'?'var(--red)':pressure==='Warning'?'var(--yellow)':'var(--dim)';
      // Pressure banner
      updatePressureBanner(pressure, avail, mem.swap_used_gb||0);
    } catch(e) {}
  }, 500);
}

function stopGenProgress() {
  if (genPollInterval) { clearInterval(genPollInterval); genPollInterval = null; }
  document.getElementById('gen-progress').classList.remove('active');
  document.getElementById('status-dot').style.background = 'var(--green)';
}

function updatePressureBanner(pressure, availGb, swapGb) {
  const banner = document.getElementById('pressure-banner');
  if (pressure === 'Critical') {
    banner.className = 'pressure-banner pressure-critical';
    let msg = '<strong>Memory pressure critical</strong> — generation may be slow. ';
    msg += 'Only ' + availGb + 'GB RAM available. ';
    if (swapGb > 0) msg += swapGb.toFixed(1) + 'GB in swap (causes slowdown). ';
    msg += 'Close other apps to free RAM, or switch to a smaller model.';
    banner.innerHTML = msg;
  } else if (pressure === 'Warning') {
    banner.className = 'pressure-banner pressure-warning';
    banner.innerHTML = '<strong>Memory pressure elevated</strong> — ' + availGb + 'GB available. '
      + 'Performance may degrade if more apps are opened.';
  } else {
    banner.className = 'pressure-banner';
  }
}

// Session persistence — survive page reload / tab switch
function saveSession() {
  try { localStorage.setItem('mlx-flash-messages', JSON.stringify(messages)); } catch(e) {}
}
function loadSession() {
  try {
    const saved = localStorage.getItem('mlx-flash-messages');
    if (saved) {
      const restored = JSON.parse(saved);
      if (restored.length > 0) {
        document.getElementById('empty')?.remove();
        restored.forEach(m => addMessage(m.role, m.content));
      }
    }
  } catch(e) {}
}
loadSession();

async function handleSlashCommand(text) {
  const cmd = text.toLowerCase().trim();

  // /clear — local only
  if (cmd === '/clear') {
    messages = [];
    localStorage.removeItem('mlx-flash-messages');
    messagesEl.innerHTML = '';
    document.getElementById('status-text').textContent = 'Conversation cleared';
    return true;
  }

  // /model <name> — use switch endpoint
  if (cmd.startsWith('/model ') && cmd !== '/models') {
    const modelName = text.slice(7).trim();
    addMessage('user', text);
    document.getElementById('status-text').textContent = 'Switching to ' + modelName + '...';
    try {
      const resp = await fetch('/v1/models/switch', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({model: modelName}),
      });
      const result = await resp.json();
      if (result.switched) {
        addMessage('assistant', '**Model switched** to `' + result.model + '`');
        document.getElementById('status-text').textContent = 'Switched to ' + result.model.split('/').pop();
      } else {
        addMessage('assistant', '**Switch failed:** ' + (result.error || 'unknown error'));
      }
    } catch(e) {
      addMessage('assistant', '**Error:** ' + e.message);
    }
    saveSession();
    return true;
  }

  // All other / commands → /commands/run
  if (cmd.startsWith('/')) {
    addMessage('user', text);
    try {
      const resp = await fetch('/commands/run', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({command: text}),
      });
      const result = await resp.json();
      const type = result.type || 'unknown';

      if (type === 'status') {
        const mem = result.memory || {};
        const w = result.workers || {};
        let md = '### Status\n\n';
        md += '| | |\n|---|---|\n';
        md += '| **Model** | `' + (result.model||'?') + '` |\n';
        md += '| **Memory** | ' + (mem.available_gb||0).toFixed(1) + 'GB available (' + (mem.pressure||'?') + ') |\n';
        if ((mem.swap_gb||0) > 0.1) md += '| **Swap** | ' + mem.swap_gb.toFixed(1) + 'GB (causes slowdown) |\n';
        md += '| **Workers** | ' + (w.healthy||0) + '/' + (w.total||0) + ' healthy, ' + (w.sessions||0) + ' sessions |\n';
        md += '| **Requests** | ' + (result.requests||0) + ' |\n';
        md += '| **Tokens** | ' + (result.tokens_generated||0) + ' |\n';
        md += '| **Uptime** | ' + Math.floor((result.uptime_s||0)/60) + 'm |\n';
        addMessage('assistant', md);
      } else if (type === 'models') {
        let md = '### Available Models\n\n';
        md += '| # | Model | Size | |\n|---|---|---|---|\n';
        (result.available||[]).forEach((m,i) => {
          const active = m.id === result.current ? ' **active**' : '';
          md += '| ' + (i+1) + ' | ' + m.label + ' | ' + m.size_gb + 'GB |' + active + ' |\n';
        });
        md += '\nSwitch: `/model <number>` or `/model <name>`';
        addMessage('assistant', md);
      } else if (type === 'help') {
        let md = '### Commands\n\n';
        md += '| Command | Description |\n|---|---|\n';
        Object.entries(result.commands||{}).forEach(([k,v]) => { md += '| `' + k + '` | ' + v + ' |\n'; });
        md += '| `/quit` | Exit the chat |\n';
        addMessage('assistant', md);
      } else if (type === 'error') {
        addMessage('assistant', '**Error:** ' + (result.error||'Unknown'));
      } else {
        addMessage('assistant', '```json\n' + JSON.stringify(result, null, 2) + '\n```');
      }
    } catch(e) {
      addMessage('assistant', '**Error:** ' + e.message);
    }
    saveSession();
    return true;
  }

  return false; // not a command
}

async function sendMessage() {
  const text = textarea.value.trim();
  if (!text || generating) return;

  textarea.value = ''; textarea.style.height = 'auto';

  // Handle slash commands
  if (text.startsWith('/')) {
    await handleSlashCommand(text);
    return;
  }

  addMessage('user', text);
  const aiEl = addMessage('assistant', '');
  generating = true;
  sendBtn.disabled = true;
  document.getElementById('status-text').textContent = 'Generating...';
  startGenProgress();

  try {
    const sel = document.getElementById('model-select');
    const currentModel = sel.value || localStorage.getItem('mlx-flash-model') || 'local';
    const payload = {
      model: currentModel,
      messages: messages.map(m => ({role: m.role, content: m.content})),
      stream: false, max_tokens: 1024,
    };

    const t0 = Date.now();
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    let fullText = '';

    if (!resp.ok) {
      let errMsg = 'Server error ' + resp.status;
      try { const j = await resp.json(); errMsg = j.error || errMsg; } catch(e) {}

      if (resp.status === 502) {
        aiEl.innerHTML = '<p style="color:var(--red)"><strong>Python worker not running</strong></p>'
          + '<p style="color:var(--dim);font-size:0.85rem;margin-top:8px">Start it in another terminal:</p>'
          + '<pre style="background:var(--surface);padding:10px 14px;border-radius:8px;margin-top:6px;font-size:0.82rem;border:1px solid var(--border)">'
          + '<code>pip install mlx mlx-lm mlx-flash\nmlx-flash --port 8081 --preload</code></pre>';
      } else {
        aiEl.innerHTML = '<p style="color:var(--red)">' + errMsg + '</p>';
      }
      messages[messages.length-1].content = 'Error: ' + errMsg;
      return;
    }

    let tokPerSec = '';
    try {
      const json = await resp.json();
      fullText = json.choices?.[0]?.message?.content
        || json.choices?.[0]?.text
        || JSON.stringify(json);
      // Extract MLX-Flash performance data if present
      const mlxData = json.mlx_flash_compress || {};
      if (mlxData.tok_per_s) tokPerSec = mlxData.tok_per_s.toFixed(1) + ' tok/s';
      const usage = json.usage || {};
      if (usage.completion_tokens && !tokPerSec) {
        tokPerSec = (usage.completion_tokens / parseFloat(elapsed)).toFixed(1) + ' tok/s';
      }
    } catch(e) {
      fullText = 'Failed to parse response';
    }

    aiEl.innerHTML = formatContent(fullText);
    messages[messages.length-1].content = fullText;
    saveSession();

    // Show generation stats in status bar
    const stats = [elapsed + 's'];
    if (tokPerSec) stats.push(tokPerSec);
    document.getElementById('status-text').textContent = 'Done — ' + stats.join(', ');
    setTimeout(() => { if (!generating) document.getElementById('status-text').textContent = 'Ready'; }, 5000);

  } catch(e) {
    aiEl.innerHTML = '<p style="color:var(--red)">Connection error — is the Python worker running?</p>';
  } finally {
    generating = false;
    sendBtn.disabled = false;
    stopGenProgress();
  }
}

// Known models catalog (matches Python MODELS list)
const KNOWN_MODELS = [
  {id:'mlx-community/gemma-4-E2B-it-4bit', label:'Gemma 4 E2B (1.5GB)', size:1.5},
  {id:'mlx-community/gemma-4-E4B-it-4bit', label:'Gemma 4 E4B (2.8GB)', size:2.8},
  {id:'mlx-community/Qwen3-4B-4bit', label:'Qwen3 4B (2.5GB)', size:2.5},
  {id:'mlx-community/Qwen3-8B-4bit', label:'Qwen3 8B (5GB)', size:5.0},
  {id:'mlx-community/gemma-4-26b-it-4bit', label:'Gemma 4 26B MoE (15GB)', size:15.0},
  {id:'mlx-community/Qwen3-30B-A3B-4bit', label:'Qwen3 30B MoE (18GB)', size:18.0},
  {id:'mlx-community/gemma-4-31b-it-4bit', label:'Gemma 4 31B (20GB)', size:20.0},
  {id:'mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit', label:'Mixtral 8x7B (26GB)', size:26.0},
];

function populateModelSelect(currentModel) {
  const sel = document.getElementById('model-select');
  const isKnown = KNOWN_MODELS.some(m => m.id === currentModel);
  let opts = '';
  if (currentModel && !isKnown) {
    const short = currentModel.split('/').pop() || currentModel;
    opts += '<option value="'+currentModel+'" selected>'+short+' (active)</option>';
  }
  opts += KNOWN_MODELS.map(m =>
    '<option value="'+m.id+'" '+(m.id===currentModel?'selected':'')+'>'+m.label+'</option>'
  ).join('');
  opts += '<option value="_custom">Custom model...</option>';
  sel.innerHTML = opts;
}

async function switchModel(modelId) {
  if (modelId === '_custom') {
    const custom = prompt('Enter HuggingFace model ID (e.g., mlx-community/gemma-4-31b-it-4bit):');
    if (!custom) return;
    modelId = custom;
  }
  const shortName = modelId.split('/').pop();
  document.getElementById('status-text').textContent = 'Switching to ' + shortName + '...';
  try {
    const resp = await fetch('/v1/models/switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: modelId}),
    });
    const result = await resp.json();
    if (result.switched) {
      document.getElementById('status-text').textContent = 'Switched to ' + shortName;
      localStorage.setItem('mlx-flash-model', modelId);
    } else if (result.error) {
      document.getElementById('status-text').textContent = 'Switch failed: ' + result.error;
    } else {
      localStorage.setItem('mlx-flash-model', modelId);
      document.getElementById('status-text').textContent = 'Model set: ' + shortName + ' (restart to apply)';
    }
  } catch(e) {
    localStorage.setItem('mlx-flash-model', modelId);
    document.getElementById('status-text').textContent = 'Model set: ' + shortName + ' (restart to apply)';
  }
}

// Poll model + memory for status bar + pressure detection
async function updateStatus() {
  try {
    const st = await fetch('/status').then(r=>r.json());
    const model = st.model || '';
    populateModelSelect(model);
    const mem = st.memory || {};
    const avail = Math.max((mem.free_gb||0)+(mem.inactive_gb||0)*0.5, 0);
    const pressure = (mem.pressure||'Normal').toString();
    document.getElementById('mem-info').textContent = avail.toFixed(1)+'GB free / '+((mem.total_gb||0).toFixed(0))+'GB';
    // Update pressure banner (visible even when not generating)
    if (!generating) updatePressureBanner(pressure, avail.toFixed(1), mem.swap_used_gb||0);
  } catch(e) {}
}
updateStatus(); setInterval(updateStatus, 5000);
</script>
</body>
</html>
"##;
