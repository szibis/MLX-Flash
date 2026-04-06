# MLX-Flash Strategy — April 2026

## Position: "The production-grade local LLM server for Mac teams"

Not competing with Ollama (simplicity) or oMLX (SSD caching). Competing on **operations**: monitoring, team management, reliability.

---

## Track 1: Close the Gaps (things competitors do better)

### 1A. One-command install (CRITICAL — highest priority)
**Gap:** Ollama = `brew install ollama`. oMLX = drag .dmg. We = clone + build + venv.
**Fix:**
- [ ] Homebrew formula: `brew tap szibis/mlx-flash && brew install mlx-flash`
- [ ] Include pre-built Rust binary in brew (universal macOS arm64)
- [ ] `mlx-flash` single command starts everything (already works with `--launch-worker`)
- [ ] Auto-install Python deps if missing

### 1B. Paged SSD KV caching (match oMLX)
**Gap:** oMLX caches KV states (computation results), we cache model weights (inputs). Theirs is more valuable.
**Fix:**
- [ ] Implement KV cache persistence to SSD (safetensors format)
- [ ] Prefix sharing: reuse cached KV when the prompt starts the same
- [ ] Two-tier: hot (RAM) + cold (SSD) with LRU eviction
- [ ] This is the biggest engineering effort — consider whether to build or integrate oMLX/vllm-mlx

### 1C. Continuous batching (match Ollama/oMLX)
**Gap:** We use process-level parallelism (N workers). They batch within one process.
**Fix:**
- [ ] Replace multi-worker with mlx-lm BatchGenerator (same as oMLX uses)
- [ ] Keep worker pool for backwards compat but default to single-process batching
- [ ] Remove GIL bottleneck via batching instead of multiple processes

### 1D. Native app / DMG (match oMLX/LM Studio)
**Gap:** We're terminal-only. They have menu bar apps.
**Fix:**
- [ ] PyObjC or SwiftUI menu bar wrapper that starts/stops the Rust server
- [ ] System tray icon showing status (green/yellow/red for pressure)
- [ ] .dmg distribution with drag-to-install
- [ ] Lower priority — terminal + web UI is fine for our target audience (engineers)

---

## Track 2: Own the Niche (things ONLY we have)

### 2A. Prometheus + Grafana stack (UNIQUE — no competitor has this)
**Status:** Done. 40+ metrics, pre-built Grafana dashboard, auto-provisioned via docker compose.
**Next:**
- [ ] Add request latency histogram: `mlx_flash_request_duration_seconds`
- [ ] Add token-to-first-token latency: `mlx_flash_ttft_seconds`
- [ ] Add per-model metrics: `mlx_flash_model_tokens_total{model="..."}`
- [ ] Grafana alerting rules in dashboard JSON

### 2B. Structured logging (UNIQUE)
**Status:** Done. JSON/text, unified fields, stdout + file.
**Next:**
- [ ] Add request_id tracing (correlate logs across Rust proxy → Python worker)
- [ ] Vector config for Loki ingestion (tested, documented)

### 2C. Team mode (UNIQUE — viral potential)
**Concept:** Shared MLX-Flash server for 3-5 developers, with per-user session isolation, usage tracking, and cost savings dashboard.
**Implementation:**
- [ ] Per-user token counting (from `user` field in OpenAI API)
- [ ] Usage dashboard: "You generated 50K tokens today, equivalent to $3.50 in API costs"
- [ ] Per-user rate limiting (optional, prevents one person hogging the GPU)
- [ ] Session isolation: each user's KV cache is separate
- [ ] `/team/stats` endpoint: per-user breakdown
- [ ] "Your team saved $X this month vs Claude API" — the viral screenshot

### 2D. Cost savings calculator (viral potential)
**Concept:** Real-time display showing "This conversation would have cost $X on Claude API".
**Implementation:**
- [ ] Track tokens per conversation
- [ ] Calculate equivalent Claude/GPT-4 API cost (use published pricing)
- [ ] Show in chat metadata bar: "128 tokens · 35 tok/s · 1.5s · **saved $0.04**"
- [ ] Dashboard: cumulative savings over time with chart
- [ ] Monthly summary: "MLX-Flash saved your team $487 in April"

---

## Track 3: Go Viral

### 3A. Brew formula (Week 1)
```ruby
class MlxFlash < Formula
  desc "Production-grade local LLM server for Mac teams"
  homepage "https://github.com/szibis/MLX-Flash"
  # ... pre-built binary + Python wheel
end
```
Test: `brew install mlx-flash && mlx-flash --model auto` starts serving in <30s.

### 3B. 30-second demo video (Week 1)
Script:
1. `brew install mlx-flash` (3s)
2. `mlx-flash` → server starts, loads model (10s)
3. Open http://localhost:8080/chat → send message → see tok/s metadata (10s)
4. Open http://localhost:8080/admin → show Grafana-style dashboard with GPU, memory, workers (5s)
5. "Free. Local. Production-grade." (2s)

### 3C. Hacker News post (Week 2)
Title: "Show HN: Production-grade LLM server for Mac teams — Prometheus metrics, auto-recovery, GPU monitoring"
Angle: "We replaced $500/month in API costs with a Mac Studio and MLX-Flash"
Include: Grafana dashboard screenshot showing real team usage

### 3D. Claude Code / Cursor integration guide (Week 2)
Title: "Free local AI coding: MLX-Flash as Claude Code backend"
- Step-by-step: install → configure MCP → code with local 30B model
- Show: real coding task completed locally at 80+ tok/s
- Compare: $20/month Claude Pro vs $0/month MLX-Flash

### 3E. Benchmarks page (Week 3)
Honest, reproducible, on real hardware:
- MLX-Flash vs Ollama vs oMLX vs LM Studio
- Same models, same hardware, same prompts
- Metrics: tok/s, TTFT, memory usage, concurrent users
- Auto-generated via CI (run weekly)

---

## Priority Order

1. **Brew install** — removes 90% of onboarding friction
2. **Cost savings calculator** — the viral screenshot ("saved $487 this month")
3. **Team mode** — the unique feature nobody else has
4. **Demo video + HN post** — distribution
5. **KV caching** — technical parity with oMLX
6. **Benchmarks** — credibility

---

## What NOT to Do

- Don't try to beat Ollama at simplicity — they have 100K stars and Docker founders
- Don't try to beat oMLX at SSD caching — they're further ahead
- Don't build an Electron app — our web UI is good enough
- Don't add features for features' sake — every addition should serve the "team ops" narrative
