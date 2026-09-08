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

## Quick start: clone → phone → chatting

On your computer (Go 1.25+):

```
git clone https://github.com/everydev1618/proxima
cd proxima
go build ./cmd/proxima
./proxima
```

On a fresh machine `proxima` prints a pairing QR and offers the starter — a
small, fast chat model (Qwen3.5 4B, ~3.6 GB). You can answer in the terminal,
or let the phone drive:

On your phone ([Expo Go](https://expo.dev/go), same WiFi — no app store build
yet):

```
cd mobile && npm install && npx expo start
```

Open it in Expo Go and scan the pairing QR — `./proxima` shows it during
first-run setup, `./proxima pair` any time after. Approve the download on the
phone, watch the star form, and when it ignites you're chatting with Iris —
every token generated on your own machine.

Already have Ollama or LM Studio running? `./proxima` uses it directly — skip
the download, run `./proxima pair`, and scan.

Headless runs never auto-download — the setup offer appears on a controlling
terminal or a paired phone, and nothing multi-gigabyte moves without a human
saying yes.

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

The mobile app (`mobile/`, Expo/React Native) drives the whole product:
first-run model download, streamed chat with Iris, and tool approvals — when
Iris wants to run a shell command, the prompt appears on the phone and the
terminal, first answer wins. The LAN listener (port 7769) is gated by a
per-machine token (`~/.vega/mobile-token`, delete it to force every phone to
re-pair); loopback stays open so the desktop dashboard is unchanged. See
`mobile/README.md`.

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
