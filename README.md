# proxima

Vega's orchestrator on **local models only**. No API keys, no config: it finds
the model server already running on your machine (Ollama, LM Studio,
llama.cpp, vLLM), picks a model, and boots Iris + Hera with tools, memory, and
the dashboard.

```
proxima
# proxima dev — ollama at http://127.0.0.1:11434, model qwen3:30b
# Dashboard: http://localhost:60726
```

No server running? On a fresh machine, `proxima` offers to set itself up: one
y at the prompt downloads a small, chatty starter model (Qwen3.5 4B, ~3.6 GB)
plus the llama.cpp runtime and drops you straight into chat. Headless runs
never auto-download — the offer only appears on a controlling terminal.

For everything past the starter, the managed runtime is driven explicitly:

```
proxima models     # the curated catalog, priced against THIS machine's memory
proxima pull       # verified llama.cpp build + the recommended model
proxima -managed   # boot the managed runtime (readiness proven by a real generation)
proxima status     # runtime, staged models, launch policies
```

Beyond the curated catalog, any of the thousands of GGUF repos on Hugging
Face can be pulled directly:

```
proxima pull hf:unsloth/Qwen3-8B-GGUF:Q4_K_M
```

The repo's file tree is listed, the chosen quant's GGUF header is streamed (a
few MB — never the weights), and the same physics check runs before anything
big downloads. Open pulls are uncurated: launch presets come from the file's
own header, and the first managed boot runs a tool-call smoke eval — the
verdict is stamped per model (`proxima status` flags failures) so you know
whether this model, at this quant, on this hardware, can actually drive an
agent.

The managed runtime is a Go port of hermes-agent's `local_runtime` (MIT,
Nous Research — see NOTICE): GGUF header parsing, VRAM/RAM budget math with a
physics refusal when a model won't fit, a 64K→96K→144K context ladder, per-
model launch policies (`presets.ini`), and a supervised `llama-server` router
with crash restarts and idle VRAM reclamation.

Flags: `-base-url` to point at a specific server (tailnet boxes count as
local), `-model` to override the model, `-managed` to skip external-server
detection, `-addr`, `-db`, `-mobile=off` to disable phone pairing.

## Phone

`proxima pair` prints a pairing QR (a first run with nothing installed shows
it on its own): scan it with the Proxima mobile app (`mobile/`, Expo/React
Native) and the phone takes over — including the first-run model download,
streamed chat with Iris, and tool approvals. The
LAN listener (port 7769) is gated by a per-machine token
(`~/.vega/mobile-token`); loopback stays open so the desktop dashboard is
unchanged. See `mobile/README.md`.

proxima is a downstream product of [govega](https://github.com/everydev1618/govega)
in the v39a-vega mold: the binary is wiring, the framework is upstream. See
`DESIGN.md` for the roadmap (managed llama.cpp runtime under OTP supervision,
hardware-aware model catalog, approval gating) and the list of upstream govega
changes it depends on.

## Build

```
go build ./cmd/proxima
go test ./...
```

A plain clone builds against the released govega. To develop against a local
`../govega` checkout, use a Go workspace: `go work init ./govega ./proxima`
from the parent directory.
