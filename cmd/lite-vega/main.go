// lite-vega boots govega's orchestrator (Iris) + builder (Hera) against a
// LOCAL model server — no API keys, no config. It probes for a running
// Ollama / LM Studio / llama.cpp / vLLM, picks a model, and serves the
// dashboard + REPL-able chat on top of it.
//
// Wiring follows the v39a-vega shape: this binary contains almost no code;
// every feature lives in govega. See DESIGN.md.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	vega "github.com/everydev1618/govega"
	"github.com/everydev1618/govega/dsl"
	"github.com/everydev1618/govega/serve"

	"github.com/everydev1618/lite-vega/localrt"
)

var version = "dev"

func main() {
	fs := flag.NewFlagSet("lite-vega", flag.ExitOnError)
	addr := fs.String("addr", "", "HTTP listen address (default: auto-assign free port)")
	dbPath := fs.String("db", vega.DefaultDBPath(), "SQLite database path")
	baseURL := fs.String("base-url", "", "local model server origin (default: auto-detect)")
	model := fs.String("model", "", "model id to use (default: first the server advertises)")
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srv, ok := resolveServer(ctx, *baseURL)
	if !ok {
		fmt.Fprint(os.Stderr, noServerHelp)
		os.Exit(1)
	}
	chosen, err := resolveModel(ctx, srv, *model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "lite-vega: %v\n%s", err, noModelHelp(srv.Type))
		os.Exit(1)
	}

	// govega's llm.New() returns the OpenAI-compat client when
	// OPENAI_BASE_URL is set; VEGA_API_KEY falls back for servers that
	// require a bearer token (llama.cpp router mode). Zero pricing is the
	// backend's default for local models.
	os.Setenv("OPENAI_BASE_URL", srv.OpenAIBaseURL())
	os.Setenv("OPENAI_MODEL", chosen)
	if os.Getenv("VEGA_API_KEY") == "" && os.Getenv("OPENAI_API_KEY") == "" {
		os.Setenv("VEGA_API_KEY", "sk-local")
	}

	doc := &dsl.Document{
		Name:      "lite-vega",
		Agents:    map[string]*dsl.Agent{},
		Workflows: map[string]*dsl.Workflow{},
	}
	interp, err := dsl.NewInterpreter(doc, dsl.WithLazySpawn())
	if err != nil {
		fmt.Fprintf(os.Stderr, "lite-vega: creating interpreter: %v\n", err)
		os.Exit(1)
	}
	defer interp.Shutdown()

	if err := vega.EnsureHome(); err != nil {
		fmt.Fprintf(os.Stderr, "lite-vega: creating vega home: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("lite-vega %s — %s at %s, model %s\n", version, srv.Type, srv.BaseURL, chosen)

	cfg := serve.Config{
		Version: version,
		Addr:    *addr,
		DBPath:  *dbPath,
		Orchestrator: dsl.IrisConfig{
			ProductName:   "Vega",
			Model:         chosen,
			FallbackModel: chosen,
		},
		Builder: dsl.HeraConfig{
			ProductName:   "Vega",
			Model:         chosen,
			FallbackModel: chosen,
		},
	}

	if err := serve.New(interp, cfg).Start(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "lite-vega: %v\n", err)
		os.Exit(1)
	}
}

// resolveServer uses the explicit -base-url when given (still identifying
// the server type behind it), otherwise probes the well-known local ports.
func resolveServer(ctx context.Context, baseURL string) (localrt.Server, bool) {
	if baseURL != "" {
		if !localrt.IsLocalEndpoint(baseURL) {
			fmt.Fprintf(os.Stderr, "warn: %s is not a local endpoint — lite-vega is built for local models\n", baseURL)
		}
		t := localrt.DetectServerType(ctx, baseURL)
		if t == localrt.ServerUnknown {
			return localrt.Server{}, false
		}
		return localrt.Server{Type: t, BaseURL: strings.TrimRight(baseURL, "/")}, true
	}
	return localrt.Detect(ctx)
}

// resolveModel returns the model to run: the explicit -model when given,
// else the first model the server advertises on /v1/models.
func resolveModel(ctx context.Context, srv localrt.Server, model string) (string, error) {
	if model != "" {
		return model, nil
	}
	models, err := localrt.ListModels(ctx, srv.OpenAIBaseURL())
	if err != nil {
		return "", fmt.Errorf("listing models on %s: %w", srv.BaseURL, err)
	}
	if len(models) == 0 {
		return "", fmt.Errorf("%s at %s has no models loaded", srv.Type, srv.BaseURL)
	}
	return models[0], nil
}

const noServerHelp = `lite-vega: no local model server found.

Checked the usual suspects:
  Ollama     http://127.0.0.1:11434
  LM Studio  http://127.0.0.1:1234
  llama.cpp  http://127.0.0.1:8080
  LiteLLM    http://127.0.0.1:4000

Start one and run lite-vega again, e.g.:
  ollama serve && ollama pull qwen3        (ollama.com)
  or open LM Studio and load a model      (lmstudio.ai)
  or llama-server -m model.gguf           (github.com/ggml-org/llama.cpp)

Or point at a server elsewhere on your network:
  lite-vega -base-url http://100.x.y.z:11434
`

func noModelHelp(t localrt.ServerType) string {
	switch t {
	case localrt.ServerOllama:
		return "Pull one first, e.g.: ollama pull qwen3\n"
	case localrt.ServerLMStudio:
		return "Load a model in LM Studio, then run lite-vega again.\n"
	default:
		return "Load a model on the server, then run lite-vega again.\n"
	}
}
