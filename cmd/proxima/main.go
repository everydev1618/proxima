// proxima boots govega's orchestrator (Iris) + builder (Hera) against a
// LOCAL model server — no API keys, no config. It uses a model server already
// running on your machine (Ollama, LM Studio, llama.cpp, vLLM), or manages
// its own llama.cpp runtime over models pulled from the curated catalog.
//
// Wiring follows the v39a-vega shape: this binary contains almost no code;
// every feature lives in govega or localrt. See DESIGN.md.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	vega "github.com/everydev1618/govega"
	"github.com/everydev1618/govega/dsl"
	"github.com/everydev1618/govega/serve"

	"github.com/everydev1618/proxima/localrt"
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
	case "pair":
		err = pairCmd(args)
	case "version":
		fmt.Println("proxima", version)
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "proxima: unknown command %q\n\n", cmd)
		printUsage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "proxima: %v\n", err)
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Print(`proxima — vega's orchestrator on local models only

Usage:
  proxima [run] [flags]   start the orchestrator (default)
  proxima models          show the model catalog with what fits THIS machine
  proxima pull [id]       install the runtime + download a model (default: recommended)
                          id can be any HF GGUF repo: hf:<org>/<repo>[:<quant>]
  proxima status          managed runtime status
  proxima pair            show the phone-pairing QR (Proxima app)
  proxima version

Run flags:
  -addr       HTTP listen address (default: auto-assign free port)
  -db         SQLite database path
  -base-url   local model server origin (default: auto-detect)
  -model      model id to use (default: first advertised)
  -managed    skip external-server detection; use the managed runtime
  -approve    tool approval gate: exec (default), all (+file writes), off
  -mobile     phone pairing: LAN listener + token gate (default true)
  -verbose    agent internals on the terminal instead of ~/.vega/logs/proxima.log
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
	approve := fs.String("approve", "exec", "tool approval gate: exec (code-running tools), all (+file writes), off")
	mobile := fs.Bool("mobile", true, "pair phones: listen on the LAN behind a pairing token")
	verbose := fs.Bool("verbose", false, "print agent internals to the terminal instead of the log file")
	if err := fs.Parse(args); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Agent internals go to a file — the terminal is for the person. The
	// phone-led first run renders a QR and a download bar; twenty INFO lines
	// between them is how a first impression dies.
	logPath := ""
	if !*verbose {
		if lf, path, err := bootLog(); err == nil {
			slog.SetDefault(slog.New(slog.NewTextHandler(lf, nil)))
			defer lf.Close()
			logPath = path
		}
	}

	// Mobile pairing: fixed LAN address + persistent token. The QR prints on
	// demand (`proxima pair`) and in the phone-led first run — not on every
	// boot.
	var setup *setupOpts
	if *mobile {
		if *addr == "" {
			*addr = defaultMobileAddr
		}
		token, err := loadOrCreatePairingToken(localrt.VegaHome())
		if err != nil {
			return err
		}
		setup = &setupOpts{addr: *addr, token: token}
	}

	endpoint, chosen, sup, err := resolveEndpoint(ctx, *baseURL, *model, *managed, setup)
	if err != nil {
		return err
	}
	if sup != nil {
		defer sup.Stop()
		// Grow-before-compact: on a context-window overflow, try granting
		// the next ladder rung (persist, bounce, prove readiness) before
		// govega compacts the conversation.
		model := chosen
		vega.SetContextPressureHook(func(p *vega.Process) bool {
			window, ok := localrt.MaybeGrowWindow(sup, model, localrt.ProbeBudget(true))
			if ok {
				fmt.Printf("context window grown to %dK for %s; retrying without compaction\n", window/1024, model)
			}
			return ok
		})
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
		Name:      "proxima",
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

	for _, line := range bootBanner(chosen, endpoint.origin) {
		fmt.Println(line)
	}

	// Tool approval gate: a local model does not run shell commands on this
	// machine without a human answering y — on this terminal or a paired
	// phone, whichever answers first.
	var apprHub *approvalHub
	if setup != nil {
		apprHub = newApprovalHub()
	}
	if gated := approvalSets[*approve]; gated != nil {
		if gate, ok := newApprover(); ok {
			gate.hub = apprHub
			interp.Tools().Use(gate.middleware(gated))
			surface := "this terminal"
			if apprHub != nil {
				surface = "this terminal or your phone"
			}
			fmt.Printf("approvals — %s ask first, on %s\n", gateNames(gated), surface)
		} else if apprHub != nil {
			interp.Tools().Use(newHubApprover(apprHub).middleware(gated))
			fmt.Println("approvals — no terminal; prompts go to paired phones")
		} else {
			fmt.Fprintln(os.Stderr, "warn: approval gate DISABLED — no controlling terminal (run with -approve=off to silence)")
		}
	} else if *approve != "off" {
		return fmt.Errorf("unknown -approve mode %q (exec, all, off)", *approve)
	}

	// The "what now" card. With a fixed addr the chat URL is known before the
	// server starts; the browser opens once it actually answers.
	chatURL := ""
	if _, port, err := net.SplitHostPort(*addr); err == nil {
		chatURL = "http://localhost:" + port
	}
	interactive := hasTTY()
	if chatURL != "" && interactive {
		openWhenReady(ctx, chatURL)
	}
	for _, line := range welcomeLines(chatURL, interactive, setup != nil, logPath) {
		fmt.Println(line)
	}

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
	if setup != nil {
		cfg.Middleware = []func(http.Handler) http.Handler{tokenMiddleware(setup.token)}
	}
	srv := serve.New(interp, cfg)
	if setup != nil {
		// The app's onboarding poller lands here once the real server is up.
		srv.RegisterRoute("GET /api/v1/local/state", readyStateHandler(chosen, endpoint.kind))
		srv.RegisterRoute("GET /api/v1/local/approvals", apprHub.handleList)
		srv.RegisterRoute("GET /api/v1/local/approvals/stream", apprHub.handleStream)
		srv.RegisterRoute("POST /api/v1/local/approvals/{id}", apprHub.handleResolve)
	}
	return srv.Start(ctx)
}

// defaultMobileAddr is the fixed mobile-mode listen address: a stable port so
// a paired phone finds proxima again after restarts. -addr overrides.
const defaultMobileAddr = "0.0.0.0:7769"

// setupOpts carries mobile-mode pairing config into the boot path.
type setupOpts struct {
	addr  string
	token string
}

// readyStateHandler answers the same /api/v1/local/state the setup server
// serves, but from the running orchestrator: setup is over, go chat.
func readyStateHandler(model, kind string) http.HandlerFunc {
	hostname, _ := os.Hostname()
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"phase":    "ready",
			"hostname": hostname,
			"model":    model,
			"server":   kind,
			"version":  version,
		})
	}
}

// pairCmd shows the pairing QR on demand. The token is per-machine and the
// mobile port is stable, so pairing works whether or not proxima is running.
func pairCmd(args []string) error {
	fs := flag.NewFlagSet("pair", flag.ExitOnError)
	addr := fs.String("addr", defaultMobileAddr, "the -addr proxima runs with")
	if err := fs.Parse(args); err != nil {
		return err
	}
	_, port, err := net.SplitHostPort(*addr)
	if err != nil {
		return fmt.Errorf("cannot parse -addr %q: %w", *addr, err)
	}
	token, err := loadOrCreatePairingToken(localrt.VegaHome())
	if err != nil {
		return err
	}
	printPairing(os.Stdout, port, token)
	return nil
}

// bootLog opens the terminal-quiet destination for agent internals.
func bootLog() (*os.File, string, error) {
	dir := filepath.Join(localrt.VegaHome(), "logs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, "", err
	}
	path := filepath.Join(dir, "proxima.log")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, "", err
	}
	return f, path, nil
}

// hasTTY reports whether a controlling terminal exists — the difference
// between a person watching and a service unit.
func hasTTY() bool {
	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		return false
	}
	tty.Close()
	return true
}

// gateNames renders a gated set for the startup banner.
func gateNames(gated map[string]bool) string {
	names := make([]string, 0, len(gated))
	for n := range gated {
		names = append(names, n)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
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
// one (caller stops it on exit). A non-nil setup routes the first-run
// bootstrap through the phone-reachable setup server.
func resolveEndpoint(ctx context.Context, baseURL, model string, forceManaged bool, setup *setupOpts) (endpoint, string, *localrt.Supervisor, error) {
	if baseURL != "" && !forceManaged {
		if !localrt.IsLocalEndpoint(baseURL) {
			fmt.Fprintf(os.Stderr, "warn: %s is not a local endpoint — proxima is built for local models\n", baseURL)
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

	// Managed runtime over staged models. An empty machine gets the
	// first-run offer: pull the starter model and go straight to chat. With
	// mobile pairing on, the offer (and the whole catalog) is answerable
	// from the phone too.
	staged := localrt.StagedModelIDs()
	if len(staged) == 0 {
		var id string
		var err error
		if setup != nil {
			id, err = bootstrapMobile(ctx, setup.addr, setup.token)
		} else {
			id, err = bootstrapStarter(ctx)
		}
		if err != nil {
			return endpoint{}, "", nil, err
		}
		if id == "" {
			return endpoint{}, "", nil, fmt.Errorf("no local model server running and no models staged.\n\n%s", noServerHelp)
		}
		staged = []string{id}
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
	// Tool-call smoke eval, once per model: readiness proves it talks, this
	// proves it drives. Advice, never a gate — a failing model still runs.
	passed, known := localrt.SmokeStamp(chosen)
	if !known {
		fmt.Printf("running tool-call smoke eval for %s...\n", chosen)
		passed = sup.TouchToolCall(chosen, 120*time.Second)
		if err := localrt.SaveSmokeStamp(chosen, passed); err != nil {
			fmt.Fprintf(os.Stderr, "warn: could not persist smoke verdict: %v\n", err)
		}
	}
	if !known || !passed {
		if passed {
			fmt.Printf("tool-call smoke: PASS — %s emits well-formed tool calls\n", chosen)
		} else {
			fmt.Fprintf(os.Stderr, "warn: tool-call smoke: FAIL — %s did not produce a valid tool call; agents may be unreliable on this model\n", chosen)
		}
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
		if e.Starter {
			note = "  [starter]"
		}
		if staged[v.ModelID()] {
			note += "  [downloaded]"
		}
		fmt.Printf("%s%-22s %5.1f GB  %s%s\n   %s\n", marker, e.ID,
			float64(e.DownloadBytes(v))/1e9, fit, note, e.Description)
	}
	if rec != nil {
		fmt.Printf("\n▸ recommended for this machine: %s (%s)\n", rec.Entry.ID, rec.Reason)
		fmt.Printf("  proxima pull %s\n", rec.Entry.ID)
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
		if repo, quant, ok := localrt.ParseHFRef(fs.Arg(0)); ok {
			return pullHF(ctx, repo, quant, *tag, budget)
		}
		entry = localrt.EntryByID(entries, fs.Arg(0))
		if entry == nil {
			return fmt.Errorf("unknown model %q — see `proxima models`, or pull any GGUF repo with hf:<org>/<repo>[:<quant>]", fs.Arg(0))
		}
	} else {
		rec := localrt.RecommendedEntry(entries, budget)
		if rec == nil {
			return fmt.Errorf("nothing in the catalog fits this machine — see `proxima models`")
		}
		entry = rec.Entry
		fmt.Printf("no model named; pulling the recommendation for this machine: %s\n", entry.ID)
	}
	choice := localrt.SelectVariant(entry, budget)
	if choice == nil {
		return fmt.Errorf("%s does not fit this machine (physics refusal) — pick a smaller model", entry.ID)
	}
	variant := choice.Variant
	if err := installRuntimeAndModel(ctx, entry, variant, *tag); err != nil {
		return err
	}
	fmt.Printf("\n%s staged. Start with: proxima -managed -model %s\n", variant.ModelID(), variant.ModelID())
	return nil
}

// pullHF is the open-catalog escape hatch: probe the repo's streamed header,
// run the same physics check as curated pulls, download only what fits.
// Nothing here is vouched for — the staged file's own header drives launch
// presets, and the first managed boot runs the tool-call smoke eval.
func pullHF(ctx context.Context, repo, quant, tag string, budget localrt.HardwareBudget) error {
	fmt.Printf("probing %s (header only, no weights)...\n", repo)
	probe, err := localrt.ProbeHFModel(ctx, repo, quant)
	if err != nil {
		return err
	}
	h := probe.Header
	fmt.Printf("  %s: %s, %d layers, %dK trained context, %.1f GB weights\n",
		probe.Variant.ModelID(), h.Architecture(), h.NLayer(), h.NCtxTrain()/1024,
		float64(probe.Profile().WeightsBytes)/1e9)
	choice := probe.Fit(budget)
	if choice == nil {
		return fmt.Errorf("%s does not fit this machine (physics refusal) — try a smaller quant", probe.Variant.ModelID())
	}
	if !choice.ZeroSpill {
		fmt.Println("  fits only spilled to RAM — expect slow decode")
	}
	if err := installRuntimeAndModel(ctx, probe.CatalogEntry(), probe.Variant, tag); err != nil {
		return err
	}
	fmt.Printf("\n%s staged (uncurated — first boot runs a tool-call check). Start with: proxima -managed -model %s\n",
		probe.Variant.ModelID(), probe.Variant.ModelID())
	return nil
}

// installRuntimeAndModel is the shared pull core: llama.cpp binaries, then
// the variant's weights + companions. Used by `proxima pull` and the
// first-run starter bootstrap.
func installRuntimeAndModel(ctx context.Context, entry *localrt.CatalogEntry, variant localrt.QuantVariant, tag string) error {
	return installModelWithProgress(ctx, entry, variant, tag, progressPrinter())
}

// installModelWithProgress is the same core with a caller-chosen progress
// sink — the phone-led bootstrap tees progress to the terminal AND the
// setup API's SSE stream.
func installModelWithProgress(ctx context.Context, entry *localrt.CatalogEntry, variant localrt.QuantVariant, tag string, prog localrt.Progress) error {
	backend := localrt.SelectBackend(localrt.DetectGPUVendor(), "")
	fmt.Printf("installing llama.cpp %s (%s)...\n", tag, backend)
	if _, err := localrt.EnsureRuntimeInstalled(tag, backend, nil, prog); err != nil {
		return fmt.Errorf("runtime install: %w", err)
	}
	fmt.Printf("downloading %s (%.1f GB) from %s...\n", variant.ModelID(),
		float64(entry.DownloadBytes(variant))/1e9, entry.Repo)
	return localrt.DownloadModel(ctx, entry, variant, prog)
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
		fmt.Println("runtime:        not installed (proxima pull)")
	} else {
		fmt.Printf("runtime:        llama.cpp %s installed\n", strings.Join(tags, ", "))
	}
	staged := localrt.StagedModelIDs()
	if len(staged) == 0 {
		fmt.Println("staged models:  none")
	} else {
		notes := make([]string, len(staged))
		for i, id := range staged {
			notes[i] = id
			if passed, known := localrt.SmokeStamp(id); known && !passed {
				notes[i] += " [tool-calls FAIL]"
			}
		}
		fmt.Printf("staged models:  %s\n", strings.Join(notes, ", "))
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

Either start one of those, or let proxima manage its own:
  proxima models        see what fits this machine
  proxima pull          download the recommended model + runtime
`

func noModelHelp(t localrt.ServerType) string {
	switch t {
	case localrt.ServerOllama:
		return "Pull one first, e.g.: ollama pull qwen3\n"
	case localrt.ServerLMStudio:
		return "Load a model in LM Studio, then run proxima again.\n"
	default:
		return "Load a model on the server, then run proxima again.\n"
	}
}
