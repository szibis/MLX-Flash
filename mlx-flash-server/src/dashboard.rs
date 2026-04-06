//! Web dashboard served at /admin with live charts for memory, cache, and performance.
//!
//! Single-page app with embedded HTML/CSS/JS — no external dependencies.
//! Polls /status and /cache/stats every 2 seconds for live data.

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
  :root { --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9; --accent: #58a6ff; --green: #3fb950; --yellow: #d29922; --red: #f85149; --dim: #484f58; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 20px; }
  h1 { font-size: 1.5rem; color: var(--accent); margin-bottom: 4px; }
  .subtitle { color: var(--dim); font-size: 0.85rem; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 20px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .card h2 { font-size: 0.9rem; color: var(--dim); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
  .metric { font-size: 2rem; font-weight: 700; }
  .metric-sm { font-size: 1.2rem; font-weight: 600; }
  .label { font-size: 0.75rem; color: var(--dim); margin-top: 4px; }
  .bar-container { height: 8px; background: var(--border); border-radius: 4px; margin-top: 8px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
  .green { color: var(--green); } .yellow { color: var(--yellow); } .red { color: var(--red); }
  .bar-green { background: var(--green); } .bar-yellow { background: var(--yellow); } .bar-red { background: var(--red); }
  canvas { width: 100% !important; height: 120px !important; }
  .chart-card { grid-column: span 2; }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .integrations { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
  .badge { background: var(--border); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  th { color: var(--dim); font-weight: 500; }
  @media (max-width: 600px) { .chart-card { grid-column: span 1; } }
</style>
</head>
<body>
<h1>MLX-Flash Dashboard</h1>
<p class="subtitle">Real-time inference monitoring &mdash; <span id="uptime">0s</span> uptime</p>

<div class="grid">
  <div class="card">
    <h2>Model</h2>
    <div class="metric-sm" id="model-name">loading...</div>
    <div class="label" id="model-info"></div>
  </div>
  <div class="card">
    <h2>Hardware</h2>
    <div class="metric-sm" id="hw-chip">detecting...</div>
    <div class="label" id="hw-ram"></div>
  </div>
  <div class="card">
    <h2>Memory Pressure</h2>
    <div class="metric" id="mem-pressure">--</div>
    <div class="bar-container"><div class="bar-fill bar-green" id="mem-bar" style="width:0%"></div></div>
    <div class="label"><span id="mem-avail">--</span> GB available / <span id="mem-total">--</span> GB</div>
  </div>
  <div class="card">
    <h2>Tokens Generated</h2>
    <div class="metric" id="tokens">0</div>
    <div class="label"><span id="requests">0</span> requests</div>
  </div>
  <div class="card">
    <h2>Cache Hit Rate</h2>
    <div class="metric" id="cache-hit">--%</div>
    <div class="bar-container"><div class="bar-fill bar-green" id="cache-bar" style="width:0%"></div></div>
    <div class="label"><span id="cache-size">0</span> experts cached</div>
  </div>
  <div class="card">
    <h2>Integrations</h2>
    <div class="integrations">
      <span class="badge">OpenAI API</span>
      <span class="badge">Ollama API</span>
      <span class="badge">MCP</span>
      <span class="badge">SSE</span>
    </div>
    <div class="label" style="margin-top:8px">Port <span id="port">8080</span> &mdash; <span id="protocol">HTTP</span></div>
  </div>

  <div class="card chart-card">
    <h2>Memory Over Time</h2>
    <canvas id="mem-chart"></canvas>
  </div>
  <div class="card chart-card">
    <h2>Tokens/sec Over Time</h2>
    <canvas id="tps-chart"></canvas>
  </div>
</div>

<div class="card" style="margin-bottom:16px">
  <h2>Optimization Hints</h2>
  <div id="hints" style="font-size:0.85rem;color:var(--dim)">Checking...</div>
</div>

<script>
const MAX_POINTS = 60;
let memHistory = [], tpsHistory = [], lastTokens = 0, lastTime = Date.now();

function drawChart(canvas, data, color, label, maxVal) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.offsetWidth * 2;
  const h = canvas.height = 240;
  ctx.clearRect(0, 0, w, h);
  if (data.length < 2) return;
  const max = maxVal || Math.max(...data, 1);
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (MAX_POINTS - 1)) * w;
    const y = h - (v / max) * (h - 20) - 10;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#484f58'; ctx.font = '20px system-ui';
  ctx.fillText(label, 8, 20);
}

async function poll() {
  try {
    const [status, cache] = await Promise.all([
      fetch('/status').then(r => r.json()),
      fetch('/cache/stats').then(r => r.json()).catch(() => null),
    ]);

    // Model + hardware
    document.getElementById('model-name').textContent = status.model || 'none';
    const mem = status.memory || {};
    document.getElementById('hw-chip').textContent = mem.chip || 'Unknown';
    document.getElementById('hw-ram').textContent = `${(mem.total_gb||0).toFixed(0)} GB unified memory`;
    document.getElementById('mem-total').textContent = (mem.total_gb||0).toFixed(0);
    document.getElementById('mem-avail').textContent = (mem.available_gb || mem.free_gb || 0).toFixed(1);

    // Memory pressure
    const pct = mem.total_gb > 0 ? ((1 - (mem.available_gb||mem.free_gb||0)/mem.total_gb) * 100) : 0;
    const pEl = document.getElementById('mem-pressure');
    pEl.textContent = pct.toFixed(0) + '%';
    pEl.className = 'metric ' + (pct > 90 ? 'red' : pct > 70 ? 'yellow' : 'green');
    const bar = document.getElementById('mem-bar');
    bar.style.width = pct + '%';
    bar.className = 'bar-fill ' + (pct > 90 ? 'bar-red' : pct > 70 ? 'bar-yellow' : 'bar-green');

    // Tokens + requests
    const stats = status.stats || {};
    const tokens = stats.tokens_generated || 0;
    document.getElementById('tokens').textContent = tokens.toLocaleString();
    document.getElementById('requests').textContent = (stats.requests||0).toLocaleString();
    document.getElementById('uptime').textContent = Math.round(stats.uptime_secs||0) + 's';

    // TPS calculation
    const now = Date.now();
    const dt = (now - lastTime) / 1000;
    const tps = dt > 0 ? (tokens - lastTokens) / dt : 0;
    lastTokens = tokens; lastTime = now;

    // Cache
    if (cache && !cache.error) {
      const hits = (cache.hot_hits||0) + (cache.warm_hits||0);
      const total = hits + (cache.cold_hits||0);
      const hitPct = total > 0 ? (hits/total*100) : 0;
      document.getElementById('cache-hit').textContent = hitPct.toFixed(0) + '%';
      document.getElementById('cache-bar').style.width = hitPct + '%';
      document.getElementById('cache-size').textContent = (cache.cached_experts||0).toLocaleString();
    }

    // Hints
    const hints = status.optimization_hints || [];
    document.getElementById('hints').innerHTML = hints.length > 0
      ? hints.map(h => '&bull; ' + h).join('<br>')
      : '<span class="green">All good — no optimization needed</span>';

    // Charts
    memHistory.push(pct); if (memHistory.length > MAX_POINTS) memHistory.shift();
    tpsHistory.push(tps); if (tpsHistory.length > MAX_POINTS) tpsHistory.shift();
    drawChart(document.getElementById('mem-chart'), memHistory, '#58a6ff', 'Memory %', 100);
    drawChart(document.getElementById('tps-chart'), tpsHistory, '#3fb950', 'tok/s', null);
  } catch(e) { /* server not ready */ }
}

poll(); setInterval(poll, 2000);
</script>
</body>
</html>
"##;
