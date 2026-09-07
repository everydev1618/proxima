# lite-vega — Design

A super-light harness around **local-only models**: vega-style tool calling,
memory, and orchestration, from download to running orchestrator with zero API
keys and zero config. Informed by a read of `govega`, `v39a.com`, and
NousResearch's `hermes-agent` (2026-09-07).

## The shape (per POSITIONING.md)

**lite-vega is a downstream product, not a fork.** Same rule as every flavor:
imports govega, never the reverse. The model to copy is v39a-vega's own README:
*"contains almost no code — the binary's job is to wire them together at
startup."* This repo is:

- `cmd/lite-vega/` — wiring in the v39a-vega shape (~1k lines when done)
- `localrt/` — the one genuinely new package: local-runtime detection and
  (later) management

Naming: the tree convention is `<flavor>-vega` (v39a-vega, apex-vega,
galley-vega), so this is **lite-vega**. Do NOT ship under "vega-lite" — that
name is taken by the well-known visualization grammar (vega.github.io/vega-lite)
and would be an SEO/identity disaster.

## Why this is 80% done already

- govega's root OTP layer (Agent/Process/Orchestrator/Supervisor, ~5.4k LOC)
  **is** the orchestrator. Nothing to build.
- Local models already work: `llm/factory.go` returns the OpenAI-compat client
  when `OPENAI_BASE_URL` is set; pricing defaults to zero ("the right answer
  for local models"). Setting `OPENAI_BASE_URL` + `OPENAI_MODEL` before
  `dsl.NewInterpreter` routes the whole stack (interpreter default LLM, serve's
  memory-extract LLM) through the local endpoint. Verified against
  `dsl/interpreter.go:119` and `serve/server.go:497`.
- The install story (pure-Go SQLite, CGO off, static cross-compiled binary,
  `//go:embed` dashboard, brew tap) is already right. Contrast: hermes-agent
  needs ~9,500 lines of bash/PowerShell plus a 5,000-line Rust Tauri installer
  to bootstrap a Python venv. A static binary makes that category of code not
  exist.

What's missing: the local path is env-var-only and undiscoverable, `vega serve`
(the CLI) hard-gates on an Anthropic key, and importing the framework drags in
`serve/` product bloat (48% of govega LOC), mssql + Docker deps, and a 40MB
binary.

## Upstream govega changes (framework-shaped — do these in govega, don't fork)

1. **Kill the API-key gate for local.** `requireAPIKey()` in `cmd/vega/serve.go`
   blocks the local story. If a local endpoint is configured or detected: no
   key, ever. (lite-vega's own main bypasses the CLI, so this doesn't block us —
   but it should be fixed for `vega serve` users too.)
2. **First-class local provider in `llm/factory.go` + `vega init`.** Init today
   prompts for Anthropic + Telegram only. It should probe for running servers
   (Ollama :11434, LM Studio :1234, llama.cpp :8080, vLLM) and write config.
   The probe logic lives here in `localrt/` first; promote it upstream once
   proven (the POSITIONING.md promotion rule).
3. **De-bloat the import graph.** Move `tools/builtin_mcp_gmail.go` and
   `builtin_mcp_mssql.go` to vega-tools (mssql is a *direct dep of the
   framework* today); gate `internal/container`'s Docker dep behind an
   interface or build tag; make Discord/Telegram gateways opt-in. Gets the
   binary from 40MB toward ~15MB.
4. **Collapse the triplicated tool loop** in `process_llm.go`
   (`executeLLMLoop` / `executeLLMStream` / `executeLLMStreamRich`), and rip
   `internal/v39a` telemetry out of the hot loop (product concern in the
   framework, wired at `process_llm.go:72-117`).
5. **Grammar-constrained tool calls.** llama.cpp's `json_schema`/GBNF support
   means the server can *guarantee* valid tool-call JSON. Small local models
   are the weak link at tool calling; this is what makes an 8B orchestrator
   reliable rather than aspirational. Add to the OpenAI-compat backend, gated
   on detected server type = llamacpp.
6. **Exact token counts for compaction.** llama.cpp exposes `/tokenize`; use it
   instead of `memory/context.go`'s 4-chars-per-token estimate when the backend
   is local. Small context windows make compaction quality matter more, not
   less.

## What lite-vega builds (product-shaped)

### Phase 1 — detect & wire (this scaffold)
- `localrt.Detect`: probe well-known local endpoints; identify server type the
  hermes way (`agent/model_metadata.py:detect_local_server_type`):
  `/api/v1/models` → LM Studio; `/api/tags` with a `models` key → Ollama (LM
  Studio 200s on that path too, hence the body check and the probe order);
  `/props` with `default_generation_settings` → llama.cpp; `/version` → vLLM.
- `localrt.IsLocalEndpoint`: loopback, RFC-1918, link-local, container-host
  DNS, **and Tailscale CGNAT 100.64/10** — a trusted Ollama box over a tailnet
  is "local" for timeout/trust purposes.
- `localrt.ListModels`: OpenAI-compat `/v1/models`.
- `cmd/lite-vega`: probe → pick model → set env → empty `dsl.Document` →
  `dsl.NewInterpreter(doc, dsl.WithLazySpawn())` → `serve.New` → Iris + Hera
  on the local model. Friendly per-server-type guidance when nothing is
  running or no model is loaded.

### Phase 2 — managed runtime (port of hermes `hermes_cli/local_runtime/`, 3,173 lines — the best code in that repo)
- Verified llama.cpp binary download per GPU backend (cuda/metal/vulkan/cpu),
  SHA256-checked, into `~/.vega/runtimes/llamacpp/<tag>/`.
- GGUF header parsing; VRAM/RAM budget probe; KV-cache byte math with a
  **PhysicsRefusal** when a model won't fit.
- Context ladder (floor 64K, grow 1.5× at 85% occupancy at turn boundaries,
  speed floor ~6 tok/s).
- **`llama-server` as an OTP child spec.** Supervise the inference engine with
  the same restart-backoff machinery vega uses for agents — readiness proven
  by an actual generation ("Reply with exactly one word: the capital of
  France."), never a health-200. This is the philosophically load-bearing
  move: vega supervising its own substrate.
- Curated `catalog.json`: a handful of GGUFs with hardware-aware variant
  selection ("you have 24GB — Qwen3-32B Q4 fits, download it?").
- Ollama honesty: query `/api/show` for `num_ctx` (its silent 2048 default
  truncates agent contexts) — port `query_ollama_num_ctx`.

### Phase 3 — harness ergonomics
- **Tool-call approval gating** — the one hermes-vs-vega.md gap worth closing
  first. An orchestrator running `exec` on your machine with an 8B model's
  judgment wants a human gate.
- Passive memory extraction moved to idle-time (extra generations are free
  locally but slow; don't block the turn).
- Terminal REPL polish (govega `repl` exists; make it the front door).

## Non-goals (deliberately not built)

No chat gateways (Discord/Telegram), no OAuth, no Postgres (SQLite only), no
peering/AIRE at v1, no control plane, no MLX-native runtime (any OpenAI-compat
MLX server works as a custom endpoint — hermes made the same call and it's
right). No port of hermes' ~9k lines of compression commit-fences and
cross-process leases — that machinery exists because cloud rate limits and
shared gateway workers exist; a single-user local box needs govega's `Compact()`
plus exact token counts.

## Download-to-gorgeous target

```
brew install everydev1618/tap/lite-vega
lite-vega
  → probes hardware + running local servers
  → LM Studio with qwen3 loaded? use it. Nothing running? offer catalog download (Phase 2)
  → readiness proven by an actual generation
  → orchestrator live: REPL + embedded dashboard, memory on, tools on
```

Zero API keys, zero config files, zero questions a non-expert can't answer.
