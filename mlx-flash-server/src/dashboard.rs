//! Web dashboard served at /admin with live charts for memory, cache, and performance.

use axum::response::Html;

pub async fn serve_dashboard() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

const DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLX-Flash Dashboard</title>
<style>
  :root {
    --bg: #0a0e14; --surface: #131920; --card: #1a2029; --border: #262d38;
    --text: #d4dce8; --dim: #5c6a7a; --accent: #4da6ff; --accent2: #7b61ff;
    --green: #2dd4a8; --yellow: #fbbf24; --red: #ef4444; --orange: #f97316;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

  .header { padding: 24px 32px 16px; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid var(--border); }
  .header h1 { font-size: 1.4rem; font-weight: 600; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .header .status { display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: var(--dim); }
  .header .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  .header .uptime { margin-left: auto; font-size: 0.8rem; color: var(--dim); font-variant-numeric: tabular-nums; }

  .container { padding: 20px 32px; }
  .grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin-bottom: 14px; }

  .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; transition: border-color 0.2s; }
  .card:hover { border-color: var(--accent); }
  .card-label { font-size: 0.7rem; font-weight: 500; color: var(--dim); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px; }
  .card-value { font-size: 2rem; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1; }
  .card-sub { font-size: 0.75rem; color: var(--dim); margin-top: 6px; }
  .card-sm { font-size: 1.1rem; font-weight: 600; }

  .span2 { grid-column: span 2; }
  .span3 { grid-column: span 3; }
  .span6 { grid-column: span 6; }

  .bar { height: 6px; background: var(--border); border-radius: 3px; margin-top: 10px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.6s cubic-bezier(0.16,1,0.3,1); }

  .green { color: var(--green); } .yellow { color: var(--yellow); } .red { color: var(--red); } .accent { color: var(--accent); }
  .bg-green { background: var(--green); } .bg-yellow { background: var(--yellow); } .bg-red { background: var(--red); } .bg-accent { background: var(--accent); }

  .chart-wrap { position: relative; height: 180px; }
  .chart-wrap canvas { position: absolute; top: 0; left: 0; width: 100% !important; height: 100% !important; }
  .chart-label { position: absolute; top: 8px; left: 12px; font-size: 0.7rem; color: var(--dim); text-transform: uppercase; letter-spacing: 0.8px; z-index: 1; }
  .chart-value { position: absolute; top: 22px; left: 12px; font-size: 1.3rem; font-weight: 700; z-index: 1; font-variant-numeric: tabular-nums; }

  .hints { display: flex; flex-direction: column; gap: 8px; }
  .hint { display: flex; align-items: flex-start; gap: 10px; padding: 10px 14px; border-radius: 10px; font-size: 0.82rem; line-height: 1.4; }
  .hint-critical { background: rgba(239,68,68,0.08); border-left: 3px solid var(--red); }
  .hint-warning { background: rgba(251,191,36,0.08); border-left: 3px solid var(--yellow); }
  .hint-info { background: rgba(77,166,255,0.08); border-left: 3px solid var(--accent); }
  .hint-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }

  .badge-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
  .badge { padding: 3px 10px; border-radius: 6px; font-size: 0.7rem; font-weight: 500; background: var(--border); }
  .badge-active { background: rgba(45,212,168,0.12); color: var(--green); border: 1px solid rgba(45,212,168,0.2); }

  @media (max-width: 900px) { .grid { grid-template-columns: repeat(2, 1fr); } .span2,.span3,.span6 { grid-column: span 2; } }
  @media (max-width: 500px) { .grid { grid-template-columns: 1fr; } .span2,.span3,.span6 { grid-column: span 1; } .container { padding: 12px 16px; } }
</style>
</head>
<body>
<div class="header">
  <h1>MLX-Flash</h1>
  <div class="status"><div class="dot"></div> Live</div>
  <div class="uptime" id="uptime">0:00</div>
</div>

<div class="container">
<div class="grid">

  <div class="card span2">
    <div class="card-label">Model</div>
    <div class="card-sm" id="model">loading...</div>
    <div class="card-sub" id="model-sub"></div>
  </div>

  <div class="card">
    <div class="card-label">Hardware</div>
    <div class="card-sm" id="hw">detecting...</div>
    <div class="card-sub" id="hw-sub"></div>
  </div>

  <div class="card">
    <div class="card-label">Memory</div>
    <div class="card-value" id="mem-pct"><span class="green">--%</span></div>
    <div class="bar"><div class="bar-fill bg-green" id="mem-bar" style="width:0%"></div></div>
    <div class="card-sub"><span id="mem-avail">--</span> / <span id="mem-total">--</span> GB</div>
  </div>

  <div class="card">
    <div class="card-label">Tokens</div>
    <div class="card-value accent" id="tokens">0</div>
    <div class="card-sub"><span id="requests">0</span> requests</div>
  </div>

  <div class="card">
    <div class="card-label">Cache Hit</div>
    <div class="card-value" id="cache-hit"><span class="green">--%</span></div>
    <div class="bar"><div class="bar-fill bg-green" id="cache-bar" style="width:0%"></div></div>
    <div class="card-sub"><span id="cache-info">no cache</span></div>
  </div>

  <div class="card span3">
    <div class="chart-wrap">
      <div class="chart-label">Memory Usage</div>
      <div class="chart-value accent" id="mem-chart-val">--%</div>
      <canvas id="mem-chart"></canvas>
    </div>
  </div>

  <div class="card span3">
    <div class="chart-wrap">
      <div class="chart-label">Tokens / sec</div>
      <div class="chart-value green" id="tps-chart-val">0</div>
      <canvas id="tps-chart"></canvas>
    </div>
  </div>

  <div class="card span3">
    <div class="card-label">Optimization Hints</div>
    <div class="hints" id="hints"></div>
  </div>

  <div class="card span3">
    <div class="card-label">Integrations</div>
    <div class="badge-row">
      <span class="badge badge-active">OpenAI API</span>
      <span class="badge badge-active">Ollama API</span>
      <span class="badge badge-active">MCP Tools</span>
      <span class="badge badge-active">SSE Streaming</span>
    </div>
    <div class="card-sub" style="margin-top:12px">Serving on port <span id="port">8080</span></div>
    <div class="badge-row" style="margin-top:8px">
      <span class="badge">Claude Code</span>
      <span class="badge">Cursor</span>
      <span class="badge">LM Studio</span>
      <span class="badge">Codex</span>
      <span class="badge">Open WebUI</span>
    </div>
  </div>

</div>
</div>

<script>
const MAX = 120;
let memH = [], tpsH = [], lastTok = 0, lastT = Date.now();

function drawChart(canvas, data, color, gradAlpha) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);
  if (data.length < 2) return;

  const max = Math.max(...data, 1) * 1.15;
  const pts = data.map((v, i) => [i / (MAX - 1) * w, h - (v / max) * (h - 50) - 4]);

  // Grid lines
  ctx.strokeStyle = 'rgba(92,106,122,0.12)'; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) { const y = h * i / 4; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

  // Gradient fill
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color.replace(')', `,${gradAlpha})`).replace('rgb', 'rgba'));
  grad.addColorStop(1, color.replace(')', ',0)').replace('rgb', 'rgba'));
  ctx.beginPath(); ctx.moveTo(pts[0][0], h);
  pts.forEach(p => ctx.lineTo(p[0], p[1]));
  ctx.lineTo(pts[pts.length-1][0], h); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  // Line
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  pts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
  ctx.stroke();

  // Current value dot
  const last = pts[pts.length - 1];
  ctx.beginPath(); ctx.arc(last[0], last[1], 4, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
  ctx.beginPath(); ctx.arc(last[0], last[1], 6, 0, Math.PI * 2);
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.globalAlpha = 0.3; ctx.stroke(); ctx.globalAlpha = 1;
}

function pressureColor(pct) { return pct > 90 ? 'red' : pct > 70 ? 'yellow' : 'green'; }
function pressureBg(pct) { return pct > 90 ? 'bg-red' : pct > 70 ? 'bg-yellow' : 'bg-green'; }
function fmtTime(s) { const m = Math.floor(s/60), sec = Math.floor(s%60); return m > 0 ? m+'m '+sec+'s' : sec+'s'; }

function renderHints(hints) {
  const el = document.getElementById('hints');
  if (!hints || hints.length === 0) {
    el.innerHTML = '<div class="hint hint-info"><span class="hint-icon">&#10003;</span>All systems nominal — no optimization needed</div>';
    return;
  }
  el.innerHTML = hints.map(h => {
    const cls = h.priority === 'critical' ? 'hint-critical' : h.priority === 'warning' ? 'hint-warning' : 'hint-info';
    const icon = h.priority === 'critical' ? '&#9888;' : h.priority === 'warning' ? '&#9888;' : '&#8505;';
    return '<div class="hint '+cls+'"><span class="hint-icon">'+icon+'</span><div>'+h.message+'</div></div>';
  }).join('');
}

async function poll() {
  try {
    const [st, cache] = await Promise.all([
      fetch('/status').then(r=>r.json()), fetch('/cache/stats').then(r=>r.json()).catch(()=>null)
    ]);
    const mem = st.memory || {}, stats = st.stats || {};

    // Model
    const mname = st.model || 'none';
    document.getElementById('model').textContent = mname.split('/').pop();
    document.getElementById('model-sub').textContent = mname;

    // Hardware
    document.getElementById('hw').textContent = (mem.total_gb||0).toFixed(0) + 'GB RAM';
    document.getElementById('hw-sub').textContent = 'Apple Silicon • ' + (mem.pressure || 'Unknown');

    // Memory
    const avail = Math.max((mem.free_gb||0) + (mem.inactive_gb||0) * 0.5, 0);
    const total = mem.total_gb || 1;
    const pct = ((1 - avail / total) * 100);
    const pc = pressureColor(pct);
    document.getElementById('mem-pct').innerHTML = '<span class="'+pc+'">'+pct.toFixed(0)+'%</span>';
    document.getElementById('mem-bar').style.width = pct+'%';
    document.getElementById('mem-bar').className = 'bar-fill '+pressureBg(pct);
    document.getElementById('mem-avail').textContent = avail.toFixed(1);
    document.getElementById('mem-total').textContent = total.toFixed(0);
    document.getElementById('mem-chart-val').innerHTML = '<span class="'+pc+'">'+pct.toFixed(0)+'%</span>';

    // Tokens
    const tok = stats.tokens_generated || 0;
    document.getElementById('tokens').textContent = tok.toLocaleString();
    document.getElementById('requests').textContent = (stats.requests||0).toLocaleString();
    document.getElementById('uptime').textContent = fmtTime(stats.uptime_secs||0);

    // TPS
    const now = Date.now(), dt = (now - lastT) / 1000;
    const tps = dt > 0 ? Math.max((tok - lastTok) / dt, 0) : 0;
    lastTok = tok; lastT = now;
    document.getElementById('tps-chart-val').textContent = tps > 0 ? tps.toFixed(1) : '0';

    // Cache
    if (cache && !cache.error) {
      const hits = (cache.hot_hits||0)+(cache.warm_hits||0);
      const t = hits+(cache.cold_hits||0);
      const hr = t > 0 ? hits/t*100 : 0;
      document.getElementById('cache-hit').innerHTML = '<span class="green">'+hr.toFixed(0)+'%</span>';
      document.getElementById('cache-bar').style.width = hr+'%';
      document.getElementById('cache-info').textContent = (cache.cached_experts||0)+' experts';
    }

    // Hints
    renderHints(st.optimization_hints);

    // Charts
    memH.push(pct); if (memH.length > MAX) memH.shift();
    tpsH.push(tps); if (tpsH.length > MAX) tpsH.shift();
    drawChart(document.getElementById('mem-chart'), memH, 'rgb(77,166,255)', 0.15);
    drawChart(document.getElementById('tps-chart'), tpsH, 'rgb(45,212,168)', 0.15);

    // Port
    document.getElementById('port').textContent = location.port || '8080';
  } catch(e) {}
}

poll(); setInterval(poll, 2000);
</script>
</body>
</html>
"##;
