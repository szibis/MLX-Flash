//! Web dashboard served at /admin with live charts, worker pool, memory breakdown, and logs.

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
  .header .nav { display: flex; gap: 8px; }
  .header .nav a { color: var(--dim); text-decoration: none; font-size: 0.8rem; padding: 4px 10px; border-radius: 6px; transition: all 0.15s; }
  .header .nav a:hover { color: var(--text); background: rgba(77,166,255,0.1); }
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

  .chart-wrap { position: relative; height: 160px; }
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
  .badge-warning { background: rgba(239,68,68,0.12); color: var(--red); border: 1px solid rgba(239,68,68,0.2); }

  .worker-row { display: flex; gap: 8px; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 0.82rem; }
  .worker-row:last-child { border-bottom: none; }
  .worker-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

  .log-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; max-height: 280px; overflow-y: auto; font-family: 'SF Mono', Menlo, monospace; font-size: 0.75rem; line-height: 1.6; }
  .log-panel::-webkit-scrollbar { width: 6px; } .log-panel::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .log-line { padding: 2px 12px; border-bottom: 1px solid rgba(38,45,56,0.3); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .log-line:hover { background: rgba(77,166,255,0.05); white-space: normal; }
  .log-info { color: var(--accent); } .log-warn { color: var(--yellow); } .log-error { color: var(--red); } .log-debug { color: var(--dim); }

  .mem-breakdown { display: flex; gap: 4px; height: 20px; border-radius: 6px; overflow: hidden; margin-top: 8px; }
  .mem-seg { height: 100%; transition: width 0.6s; position: relative; }
  .mem-seg:hover::after { content: attr(data-label); position: absolute; top: -24px; left: 50%; transform: translateX(-50%); font-size: 0.65rem; color: var(--text); background: var(--card); padding: 2px 6px; border-radius: 4px; white-space: nowrap; border: 1px solid var(--border); }

  @media (max-width: 900px) { .grid { grid-template-columns: repeat(2, 1fr); } .span2,.span3,.span6 { grid-column: span 2; } }
  @media (max-width: 500px) { .grid { grid-template-columns: 1fr; } .span2,.span3,.span6 { grid-column: span 1; } .container { padding: 12px 16px; } }
</style>
</head>
<body>
<div class="header">
  <h1>MLX-Flash</h1>
  <div class="status"><div class="dot"></div> Live</div>
  <div class="nav"><a href="/admin">Dashboard</a><a href="/chat">Chat</a><a href="/metrics">Metrics</a></div>
  <div class="uptime" id="uptime">0:00</div>
</div>

<div class="container">
<div class="grid">

  <!-- Row 1: Key stats -->
  <div class="card span2">
    <div class="card-label">Model</div>
    <div class="card-sm" id="model">loading...</div>
    <div class="card-sub" id="model-sub"></div>
  </div>
  <div class="card">
    <div class="card-label">Memory</div>
    <div class="card-value" id="mem-pct"><span class="green">--%</span></div>
    <div class="bar"><div class="bar-fill bg-green" id="mem-bar" style="width:0%"></div></div>
    <div class="card-sub"><span id="mem-avail">--</span> / <span id="mem-total">--</span> GB</div>
  </div>
  <div class="card">
    <div class="card-label">Requests</div>
    <div class="card-value accent" id="requests">0</div>
    <div class="card-sub"><span id="req-rate">0</span> req/s</div>
  </div>
  <div class="card">
    <div class="card-label">Tokens</div>
    <div class="card-value" id="tokens"><span class="accent">0</span></div>
    <div class="card-sub"><span id="tok-rate">0</span> tok/s</div>
  </div>
  <div class="card">
    <div class="card-label">Pressure</div>
    <div class="card-value" id="pressure"><span class="green">OK</span></div>
    <div class="card-sub">Swap: <span id="swap">0</span> GB</div>
  </div>

  <!-- Row 2: Charts -->
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

  <!-- Row 3: Memory breakdown + Workers -->
  <div class="card span3">
    <div class="card-label">Memory Breakdown</div>
    <div class="mem-breakdown" id="mem-breakdown"></div>
    <div class="card-sub" style="margin-top:10px">
      <span style="color:var(--red)">&#9632;</span> Active
      <span style="color:var(--orange)">&#9632;</span> Wired
      <span style="color:var(--yellow)">&#9632;</span> Compressed
      <span style="color:var(--dim)">&#9632;</span> Inactive
      <span style="color:var(--green)">&#9632;</span> Free
    </div>
    <div class="card-sub" id="mem-detail"></div>
  </div>
  <div class="card span3">
    <div class="card-label">Workers <span id="worker-summary" style="float:right;color:var(--green)"></span></div>
    <div id="workers"></div>
    <div class="card-sub" style="margin-top:8px">Sessions: <span id="sessions">0</span> | Strategy: least-connections + cache-affinity</div>
  </div>

  <!-- Row 4: Hints + Cache -->
  <div class="card span3">
    <div class="card-label">Optimization Hints</div>
    <div class="hints" id="hints"></div>
  </div>
  <div class="card span3">
    <div class="card-label">Cache</div>
    <div style="display:flex;gap:20px;align-items:baseline">
      <div><span class="card-value" id="cache-hit" style="font-size:1.5rem"><span class="green">--%</span></span><div class="card-sub">hit rate</div></div>
      <div><span class="card-sm" id="cache-entries" style="color:var(--orange)">0</span><div class="card-sub">entries</div></div>
    </div>
    <div class="bar"><div class="bar-fill bg-green" id="cache-bar" style="width:0%"></div></div>
  </div>

  <!-- Row 5: Live Logs -->
  <div class="card span6">
    <div class="card-label">Live Logs <span style="float:right;font-weight:400;text-transform:none;letter-spacing:0">last 100 entries</span></div>
    <div class="log-panel" id="log-panel"></div>
  </div>

</div>
</div>

<script>
const MAX = 120;
let memH = [], tpsH = [], rpsH = [], lastTok = 0, lastReq = 0, lastT = Date.now();

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
  ctx.strokeStyle = 'rgba(92,106,122,0.12)'; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) { const y = h * i / 4; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color.replace(')', `,${gradAlpha})`).replace('rgb', 'rgba'));
  grad.addColorStop(1, color.replace(')', ',0)').replace('rgb', 'rgba'));
  ctx.beginPath(); ctx.moveTo(pts[0][0], h);
  pts.forEach(p => ctx.lineTo(p[0], p[1]));
  ctx.lineTo(pts[pts.length-1][0], h); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  pts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
  ctx.stroke();
  const last = pts[pts.length - 1];
  ctx.beginPath(); ctx.arc(last[0], last[1], 4, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
}

function pressureColor(pct) { return pct > 90 ? 'red' : pct > 70 ? 'yellow' : 'green'; }
function pressureBg(pct) { return pct > 90 ? 'bg-red' : pct > 70 ? 'bg-yellow' : 'bg-green'; }
function fmtTime(s) { const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = Math.floor(s%60); return h > 0 ? h+'h '+m+'m' : m > 0 ? m+'m '+sec+'s' : sec+'s'; }

function renderHints(hints) {
  const el = document.getElementById('hints');
  if (!hints || hints.length === 0) {
    el.innerHTML = '<div class="hint hint-info"><span class="hint-icon">&#10003;</span>All systems nominal</div>';
    return;
  }
  el.innerHTML = hints.map(h => {
    const cls = h.priority === 'critical' ? 'hint-critical' : h.priority === 'warning' ? 'hint-warning' : 'hint-info';
    const icon = h.priority === 'critical' ? '&#9888;' : h.priority === 'warning' ? '&#9888;' : '&#8505;';
    return '<div class="hint '+cls+'"><span class="hint-icon">'+icon+'</span><div>'+h.message+'</div></div>';
  }).join('');
}

function renderWorkers(wdata) {
  const el = document.getElementById('workers');
  if (!wdata || !wdata.workers) { el.innerHTML = '<div class="card-sub">No worker data</div>'; return; }
  el.innerHTML = wdata.workers.map(w => {
    const dotColor = w.healthy ? 'var(--green)' : 'var(--red)';
    const status = w.healthy ? 'healthy' : 'down';
    return '<div class="worker-row">'
      + '<div class="worker-dot" style="background:'+dotColor+'"></div>'
      + '<span>:'+w.port+'</span>'
      + '<span style="color:var(--dim);margin-left:auto">'+w.inflight+' inflight</span>'
      + '<span style="color:var(--dim);margin-left:12px">'+w.total_requests+' total</span>'
      + '</div>';
  }).join('');
  document.getElementById('worker-summary').textContent = wdata.healthy_count + '/' + wdata.total_count + ' healthy';
}

function renderMemBreakdown(mem) {
  const el = document.getElementById('mem-breakdown');
  const total = mem.total_gb || 1;
  const pct = v => ((v||0) / total * 100).toFixed(1);
  el.innerHTML = [
    {v: mem.active_gb, c: 'var(--red)', l: 'Active '+pct(mem.active_gb)+'%'},
    {v: mem.wired_gb, c: 'var(--orange)', l: 'Wired '+pct(mem.wired_gb)+'%'},
    {v: mem.compressed_gb, c: 'var(--yellow)', l: 'Compressed '+pct(mem.compressed_gb)+'%'},
    {v: mem.inactive_gb, c: 'var(--dim)', l: 'Inactive '+pct(mem.inactive_gb)+'%'},
    {v: mem.free_gb, c: 'var(--green)', l: 'Free '+pct(mem.free_gb)+'%'},
  ].map(s => '<div class="mem-seg" style="width:'+pct(s.v)+'%;background:'+s.c+'" data-label="'+s.l+'"></div>').join('');
  document.getElementById('mem-detail').textContent =
    'Active: '+(mem.active_gb||0).toFixed(1)+'G | Wired: '+(mem.wired_gb||0).toFixed(1)+'G | Compressed: '+(mem.compressed_gb||0).toFixed(1)+'G | Inactive: '+(mem.inactive_gb||0).toFixed(1)+'G | Free: '+(mem.free_gb||0).toFixed(1)+'G';
}

function renderLogs(logs) {
  const el = document.getElementById('log-panel');
  if (!logs || logs.length === 0) { el.innerHTML = '<div class="log-line" style="color:var(--dim)">No logs yet</div>'; return; }
  const wasAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
  el.innerHTML = logs.reverse().map(l => {
    const cls = l.level === 'error' ? 'log-error' : l.level === 'warn' || l.level === 'warning' ? 'log-warn' : l.level === 'debug' ? 'log-debug' : 'log-info';
    const ts = l.timestamp || '';
    return '<div class="log-line"><span class="'+cls+'">['+l.level.toUpperCase().padEnd(5)+']</span> <span style="color:var(--dim)">'+l.component+'</span> '+l.message+'</div>';
  }).join('');
  if (wasAtBottom) el.scrollTop = el.scrollHeight;
}

async function poll() {
  try {
    const [st, cache, workers, logs] = await Promise.all([
      fetch('/status').then(r=>r.json()),
      fetch('/cache/stats').then(r=>r.json()).catch(()=>null),
      fetch('/workers').then(r=>r.json()).catch(()=>null),
      fetch('/logs/recent').then(r=>r.json()).catch(()=>null),
    ]);
    const mem = st.memory || {}, stats = st.stats || {};

    // Model
    const mname = st.model || 'none';
    document.getElementById('model').textContent = mname.split('/').pop();
    document.getElementById('model-sub').textContent = mname;

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

    // Pressure + Swap
    const pressure = (mem.pressure||'Normal').toString();
    const pClass = pressure === 'Critical' ? 'red' : pressure === 'Warning' ? 'yellow' : 'green';
    document.getElementById('pressure').innerHTML = '<span class="'+pClass+'">'+pressure+'</span>';
    document.getElementById('swap').textContent = (mem.swap_used_gb||0).toFixed(1);

    // Requests + Tokens + Rates
    const tok = stats.tokens_generated || 0;
    const req = stats.requests || 0;
    const now = Date.now(), dt = (now - lastT) / 1000;
    const tps = dt > 0 ? Math.max((tok - lastTok) / dt, 0) : 0;
    const rps = dt > 0 ? Math.max((req - lastReq) / dt, 0) : 0;
    lastTok = tok; lastReq = req; lastT = now;
    document.getElementById('tokens').innerHTML = '<span class="accent">'+tok.toLocaleString()+'</span>';
    document.getElementById('requests').textContent = req.toLocaleString();
    document.getElementById('tok-rate').textContent = tps > 0 ? tps.toFixed(1) : '0';
    document.getElementById('req-rate').textContent = rps > 0 ? rps.toFixed(2) : '0';
    document.getElementById('tps-chart-val').textContent = tps > 0 ? tps.toFixed(1) : '0';
    document.getElementById('uptime').textContent = fmtTime(stats.uptime_secs||0);

    // Cache
    if (cache && !cache.error) {
      const hits = (cache.hot_hits||0)+(cache.warm_hits||0);
      const t = hits+(cache.cold_hits||0);
      const hr = t > 0 ? hits/t*100 : 0;
      document.getElementById('cache-hit').innerHTML = '<span class="green">'+hr.toFixed(0)+'%</span>';
      document.getElementById('cache-bar').style.width = hr+'%';
      document.getElementById('cache-entries').textContent = (cache.cached_experts||0);
    }

    // Hints
    renderHints(st.optimization_hints);

    // Workers
    renderWorkers(workers || st.workers);
    if (workers) document.getElementById('sessions').textContent = workers.sessions_active || st.workers?.sessions_active || '0';

    // Memory breakdown
    renderMemBreakdown(mem);

    // Logs
    if (logs && logs.logs) renderLogs(logs.logs);

    // Charts
    memH.push(pct); if (memH.length > MAX) memH.shift();
    tpsH.push(tps); if (tpsH.length > MAX) tpsH.shift();
    drawChart(document.getElementById('mem-chart'), memH, 'rgb(77,166,255)', 0.15);
    drawChart(document.getElementById('tps-chart'), tpsH, 'rgb(45,212,168)', 0.15);
  } catch(e) {}
}

poll(); setInterval(poll, 2000);
</script>
</body>
</html>
"##;
