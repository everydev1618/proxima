# lite-vega

Vega's orchestrator on **local models only**. No API keys, no config: it finds
the model server already running on your machine (Ollama, LM Studio,
llama.cpp, vLLM), picks a model, and boots Iris + Hera with tools, memory, and
the dashboard.

```
lite-vega
# lite-vega dev — ollama at http://127.0.0.1:11434, model qwen3:30b
# Dashboard: http://localhost:60726
```

Flags: `-base-url` to point at a specific server (tailnet boxes count as
local), `-model` to override the model, `-addr`, `-db`.

lite-vega is a downstream product of [govega](https://github.com/everydev1618/govega)
in the v39a-vega mold: the binary is wiring, the framework is upstream. See
`DESIGN.md` for the roadmap (managed llama.cpp runtime under OTP supervision,
hardware-aware model catalog, approval gating) and the list of upstream govega
changes it depends on.

## Build

```
go build ./cmd/lite-vega
go test ./...
```

Requires `../govega` checked out as a sibling (module `replace` directive).
