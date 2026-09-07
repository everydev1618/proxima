// lite-vega boots govega's orchestrator (Iris) + builder (Hera) against a
// LOCAL model server — no API keys, no config. It uses a model server already
// running on your machine (Ollama, LM Studio, llama.cpp, vLLM), or manages
// its own llama.cpp runtime over models pulled from the curated catalog.
//
// Wiring follows the v39a-vega shape: this binary contains almost no code;
// every feature lives in govega or localrt. See DESIGN.md.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	vega "github.com/everydev1618/govega"
	"github.com/everydev1618/govega/dsl"
	"github.com/everydev1618/govega/serve"

	"github.com/everydev1618/lite-vega/localrt"
)

var version = "dev"

func main() {
	args := os.Args[1:]
	cmd := "run"
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		cmd, args = args[0], args[1:]
	}
	var err error
	switch cmd {
	case "run":
		err = runCmd(args)
	case "models":
		err = modelsCmd(args)
	case "pull":
		err = pullCmd(args)
	case "status":
		err = statusCmd(args)
	case "version":
		fmt.Println("lite-vega", version)
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "lite-vega: unknown command %q\n\n", cmd)
		printUsage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "lite-vega: %v\n", err)
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Print(`lite-vega — vega's orchestrator on local models only

Usage:
  lite-vega [run] [flags]   start the orchestrator (default)
  lite-vega models          show the model catalog with what fits THIS machine
  lite-vega pull [id]       install the runtime + download a model (default: recommended)
  lite-vega status          managed runtime status
  lite-vega version

Run flags:
  -addr       HTTP listen address (default: auto-assign free port)
  -db         SQLite database path
  -base-url   local model server origin (default: auto-detect)
  -model      model id to use (default: first advertised)
  -managed    skip external-server detection; use the managed runtime
`)
}

// ── run ──────────────────────────────────────────────────────

func runCmd(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	addr := fs.String("addr", "", "HTTP listen address")
	dbPath := fs.String("db", vega.DefaultDBPath(), "SQLite database path")
	baseURL := fs.String("base-url", "", "local model server origin")
	model := fs.String("model", "", "model id to use")
	managed := fs.Bool("managed", false, "use the managed runtime even when an external server runs")
	if err := fs.Parse(args); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	endpoint, chosen, sup, err := resolveEndpoint(ctx, *baseURL, *model, *managed)
	if err != nil {
		return err
	}
	if sup != nil {
		defer sup.Stop()
	}

	// govega's llm.New() returns the OpenAI-compat client when
	// OPENAI_BASE_URL is set. Zero pricing is the backend's default for
	// local models.
	os.Setenv("OPENAI_BASE_URL", endpoint.baseURL)
	os.Setenv("OPENAI_MODEL", chosen)
	if endpoint.apiKey != "" {
		os.Setenv("VEGA_API_KEY", endpoint.apiKey)
	} else if os.Getenv("VEGA_API_KEY") == "" && os.Getenv("OPENAI_API_KEY") == "" {
		os.Setenv("VEGA_API_KEY", "sk-local")
	}

	doc := &dsl.Document{
		Name:      "lite-vega",
		Agents:    map[string]*dsl.Agent{},
		Workflows: map[string]*dsl.Workflow{},
	}
	interp, err := dsl.NewInterpreter(doc, dsl.WithLazySpawn())
	if err != nil {
		return fmt.Errorf("creating interpreter: %w", err)
	}
	defer interp.Shutdown()
	if err := vega.EnsureHome(); err != nil {
		return fmt.Errorf("creating vega home: %w", err)
	}

	fmt.Printf("lite-vega %s — %s at %s, model %s\n", version, endpoint.kind, endpoint.origin, chosen)

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
	return serve.New(interp, cfg).Start(ctx)
}

type endpoint struct {
	kind    string // server type label for the banner
	origin  string // server origin (no /v1)
	baseURL string // OpenAI-compat root (with /v1)
	apiKey  string
}

// resolveEndpoint picks, in order: the explicit -base-url, a detected
// external server, the managed runtime over staged models. It returns the
// endpoint, the model id, and the supervisor when the managed path booted
// one (caller stops it on exit).
func resolveEndpoint(ctx context.Context, baseURL, model string, forceManaged bool) (endpoint, string, *localrt.Supervisor, error) {
	if baseURL != "" && !forceManaged {
		if !localrt.IsLocalEndpoint(baseURL) {
			fmt.Fprintf(os.Stderr, "warn: %s is not a local endpoint — lite-vega is built for local models\n", baseURL)
		}
		t := localrt.DetectServerType(ctx, baseURL)
		if t == localrt.ServerUnknown {
			return endpoint{}, "", nil, fmt.Errorf("no recognizable model server at %s", baseURL)
		}
		srv := localrt.Server{Type: t, BaseURL: strings.TrimRight(baseURL, "/")}
		chosen, err := pickExternalModel(ctx, srv, model)
		if err != nil {
			return endpoint{}, "", nil, err
		}
		return endpoint{kind: string(t), origin: srv.BaseURL, baseURL: srv.OpenAIBaseURL()}, chosen, nil, nil
	}

	if !forceManaged {
		if srv, ok := localrt.Detect(ctx); ok {
			chosen, err := pickExternalModel(ctx, srv, model)
			if err != nil {
				return endpoint{}, "", nil, err
			}
			return endpoint{kind: string(srv.Type), origin: srv.BaseURL, baseURL: srv.OpenAIBaseURL()}, chosen, nil, nil
		}
	}

	// Managed runtime over staged models.
	staged := localrt.StagedModelIDs()
	if len(staged) == 0 {
		return endpoint{}, "", nil, fmt.Errorf("no local model server running and no models staged.\n\n%s", noServerHelp)
	}
	sup, err := localrt.EnsureManagedRuntime("")
	if err != nil {
		return endpoint{}, "", nil, err
	}
	chosen := model
	if chosen == "" {
		chosen = staged[0]
	}
	fmt.Printf("loading %s (first token after a cold load is slow)...\n", chosen)
	sup.SetPrimaryModel(chosen)
	ready, err := sup.EnsureModelReady(chosen, 600*time.Second)
	if err != nil {
		sup.Stop()
		return endpoint{}, "", nil, err
	}
	if !ready {
		sup.Stop()
		return endpoint{}, "", nil, fmt.Errorf("model %s loaded but failed the readiness generation (log: %s)", chosen, sup.LogPath)
	}
	return endpoint{kind: "managed llama.cpp", origin: fmt.Sprintf("http://127.0.0.1:%d", sup.Port),
		baseURL: sup.BaseURL(), apiKey: sup.APIKey}, chosen, sup, nil
}

// pickExternalModel returns the model to run on an external server: the
// explicit -model, else the first advertised. Warns when Ollama would
// silently truncate the context.
func pickExternalModel(ctx context.Context, srv localrt.Server, model string) (string, error) {
	if model == "" {
		models, err := localrt.ListModels(ctx, srv.OpenAIBaseURL())
		if err != nil {
			return "", fmt.Errorf("listing models on %s: %w", srv.BaseURL, err)
		}
		if len(models) == 0 {
			return "", fmt.Errorf("%s at %s has no models loaded.\n%s", srv.Type, srv.BaseURL, noModelHelp(srv.Type))
		}
		model = models[0]
	}
	if srv.Type == localrt.ServerOllama {
		if numCtx := localrt.QueryOllamaNumCtx(ctx, srv.BaseURL, model); numCtx > 0 && numCtx < 32768 {
			fmt.Fprintf(os.Stderr,
				"warn: ollama serves %s with num_ctx=%d — agent contexts will silently truncate.\n"+
					"      Fix: OLLAMA_CONTEXT_LENGTH=65536 ollama serve, or a Modelfile with PARAMETER num_ctx 65536\n",
				model, numCtx)
		}
	}
	return model, nil
}

// ── models ───────────────────────────────────────────────────

func modelsCmd(args []string) error {
	fs := flag.NewFlagSet("models", flag.ExitOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	budget := localrt.ProbeBudget(true)
	entries := localrt.Catalog()
	rec := localrt.RecommendedEntry(entries, budget)
	staged := map[string]bool{}
	for _, id := range localrt.StagedModelIDs() {
		staged[id] = true
	}

	fmt.Printf("Machine budget: %.0f GiB usable", float64(budget.UsableVRAMBytes)/(1<<30))
	if budget.UMA {
		fmt.Print(" (unified memory)")
	}
	fmt.Print("\n\n")
	for _, e := range entries {
		v := e.Variants[len(e.Variants)-1]
		marker := "  "
		if rec != nil && rec.Entry.ID == e.ID {
			marker = "▸ "
		}
		fit := "does not fit (needs a smaller quant)"
		if c := localrt.SelectVariant(e, budget); c != nil {
			switch c.ReasonKey {
			case "best-large-window":
				fit = "fits with a large context window"
			case "best-fits":
				fit = "fits at the 64K floor"
			case "smallest-fits-spilled":
				fit = "runs spilled to RAM (slow)"
			}
			tokS := localrt.PredictedDecodeTokS(e, c.Variant, budget, !c.ZeroSpill)
			fit += fmt.Sprintf(", ~%.0f tok/s", tokS)
		}
		note := ""
		if staged[v.ModelID()] {
			note = "  [downloaded]"
		}
		fmt.Printf("%s%-22s %5.1f GB  %s%s\n   %s\n", marker, e.ID,
			float64(e.DownloadBytes(v))/1e9, fit, note, e.Description)
	}
	if rec != nil {
		fmt.Printf("\n▸ recommended for this machine: %s (%s)\n", rec.Entry.ID, rec.Reason)
		fmt.Printf("  lite-vega pull %s\n", rec.Entry.ID)
	} else {
		fmt.Println("\nNothing in the catalog fits this machine.")
	}
	return nil
}

// ── pull ─────────────────────────────────────────────────────

func pullCmd(args []string) error {
	fs := flag.NewFlagSet("pull", flag.ExitOnError)
	tag := fs.String("runtime-tag", localrt.DefaultRuntimeTag, "llama.cpp release tag")
	if err := fs.Parse(args); err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	budget := localrt.ProbeBudget(true)
	entries := localrt.Catalog()
	var entry *localrt.CatalogEntry
	if fs.NArg() > 0 {
		entry = localrt.EntryByID(entries, fs.Arg(0))
		if entry == nil {
			return fmt.Errorf("unknown model %q — see `lite-vega models`", fs.Arg(0))
		}
	} else {
		rec := localrt.RecommendedEntry(entries, budget)
		if rec == nil {
			return fmt.Errorf("nothing in the catalog fits this machine — see `lite-vega models`")
		}
		entry = rec.Entry
		fmt.Printf("no model named; pulling the recommendation for this machine: %s\n", entry.ID)
	}
	choice := localrt.SelectVariant(entry, budget)
	if choice == nil {
		return fmt.Errorf("%s does not fit this machine (physics refusal) — pick a smaller model", entry.ID)
	}
	variant := choice.Variant

	// 1. Runtime binaries.
	backend := localrt.SelectBackend(localrt.DetectGPUVendor(), "")
	fmt.Printf("installing llama.cpp %s (%s)...\n", *tag, backend)
	if _, err := localrt.EnsureRuntimeInstalled(*tag, backend, nil, progressPrinter()); err != nil {
		return fmt.Errorf("runtime install: %w", err)
	}

	// 2. Model weights (+ companions).
	fmt.Printf("downloading %s (%.1f GB) from %s...\n", variant.ModelID(),
		float64(entry.DownloadBytes(variant))/1e9, entry.Repo)
	if err := localrt.DownloadModel(ctx, entry, variant, progressPrinter()); err != nil {
		return err
	}
	fmt.Printf("\n%s staged. Start with: lite-vega -managed -model %s\n", variant.ModelID(), variant.ModelID())
	return nil
}

// progressPrinter renders one carriage-return progress line per stage/label.
func progressPrinter() localrt.Progress {
	var lastLine int
	return func(stage string, done, total int64, label string) {
		var line string
		if total > 0 {
			line = fmt.Sprintf("  %s %s %3d%% (%.1f/%.1f GB)", stage, label,
				done*100/total, float64(done)/1e9, float64(total)/1e9)
		} else {
			line = fmt.Sprintf("  %s %s %.1f GB", stage, label, float64(done)/1e9)
		}
		pad := lastLine - len(line)
		if pad < 0 {
			pad = 0
		}
		fmt.Printf("\r%s%s", line, strings.Repeat(" ", pad))
		lastLine = len(line)
		if total > 0 && done >= total {
			fmt.Println()
			lastLine = 0
		}
	}
}

// ── status ───────────────────────────────────────────────────

func statusCmd(args []string) error {
	fmt.Printf("vega home:      %s\n", localrt.VegaHome())
	tags := localrt.InstalledTags()
	if len(tags) == 0 {
		fmt.Println("runtime:        not installed (lite-vega pull)")
	} else {
		fmt.Printf("runtime:        llama.cpp %s installed\n", strings.Join(tags, ", "))
	}
	staged := localrt.StagedModelIDs()
	if len(staged) == 0 {
		fmt.Println("staged models:  none")
	} else {
		fmt.Printf("staged models:  %s\n", strings.Join(staged, ", "))
	}
	if state := localrt.ReadServerState(); state != nil {
		fmt.Printf("managed server: %s (pid %d)\n", state.BaseURL, state.PID)
	} else {
		fmt.Println("managed server: not running")
	}
	if decisions := localrt.ReadPresetDecisions(filepath.Join(localrt.RuntimesRoot(), "presets.ini")); len(decisions) > 0 {
		for id, d := range decisions {
			spill := ""
			if d.Spilled {
				spill = " (spilled)"
			}
			fmt.Printf("  policy %-30s window %dK%s\n", id, d.Window/1024, spill)
		}
	}
	if srv, ok := localrt.Detect(context.Background()); ok {
		fmt.Printf("external:       %s at %s\n", srv.Type, srv.BaseURL)
	}
	return nil
}

const noServerHelp = `Checked the usual suspects:
  Ollama     http://127.0.0.1:11434
  LM Studio  http://127.0.0.1:1234
  llama.cpp  http://127.0.0.1:8080
  LiteLLM    http://127.0.0.1:4000

Either start one of those, or let lite-vega manage its own:
  lite-vega models        see what fits this machine
  lite-vega pull          download the recommended model + runtime
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
