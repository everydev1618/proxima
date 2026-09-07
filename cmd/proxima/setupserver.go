// Phone-led first-run: before any model exists, proxima serves a tiny
// token-gated HTTP API on the final port so the mobile app can show the
// catalog priced for THIS machine, take the download consent, and stream
// progress. When the model is ready the setup server hands the port to the
// real govega serve — the app polls its way from "downloading" into chat.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/everydev1618/proxima-vega/localrt"
)

type setupPhase string

const (
	phaseSetup       setupPhase = "setup"
	phaseDownloading setupPhase = "downloading"
	phaseStarting    setupPhase = "starting"
	phaseError       setupPhase = "error"
)

// setupModel is one catalog row priced against this machine.
type setupModel struct {
	ID          string  `json:"id"`
	DisplayName string  `json:"display_name"`
	Description string  `json:"description,omitempty"`
	DownloadGB  float64 `json:"download_gb"`
	Fits        bool    `json:"fits"`
	FitReason   string  `json:"fit_reason,omitempty"`
	TokS        float64 `json:"tok_s,omitempty"`
	Starter     bool    `json:"starter,omitempty"`
	Staged      bool    `json:"staged,omitempty"`
}

// setupInfo is the immutable part of the setup state: the machine and what
// it can run.
type setupInfo struct {
	Hostname    string
	UsableGB    float64
	UMA         bool
	Recommended string
	Models      []setupModel
}

// progressEvent mirrors localrt.Progress calls onto the wire.
type progressEvent struct {
	Type  string `json:"type"` // "progress"
	Stage string `json:"stage"`
	Label string `json:"label"`
	Done  int64  `json:"done"`
	Total int64  `json:"total"`
}

// setupHub holds mutable setup state and fans events out to SSE subscribers.
type setupHub struct {
	info setupInfo

	mu        sync.Mutex
	phase     setupPhase
	errMsg    string
	prog      progressEvent
	consented bool
	subs      map[chan []byte]struct{}

	// consent delivers the chosen model id exactly once — whoever answers
	// first (phone POST or terminal y) wins.
	consent chan string
}

func newSetupHub(info setupInfo) *setupHub {
	return &setupHub{
		info:    info,
		phase:   phaseSetup,
		subs:    map[chan []byte]struct{}{},
		consent: make(chan string, 1),
	}
}

// offerConsent records a consent decision; false when someone already won.
func (h *setupHub) offerConsent(modelID string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.consented {
		return false
	}
	h.consented = true
	h.consent <- modelID
	return true
}

func (h *setupHub) setPhase(p setupPhase) {
	h.mu.Lock()
	h.phase = p
	h.mu.Unlock()
	h.broadcast(map[string]any{"type": "phase", "phase": p})
}

func (h *setupHub) setError(msg string) {
	h.mu.Lock()
	h.phase = phaseError
	h.errMsg = msg
	h.mu.Unlock()
	h.broadcast(map[string]any{"type": "phase", "phase": phaseError, "error": msg})
}

// progressFn adapts the hub into the localrt.Progress callback the
// downloader already speaks.
func (h *setupHub) progressFn() localrt.Progress {
	var last time.Time
	return func(stage string, done, total int64, label string) {
		ev := progressEvent{Type: "progress", Stage: stage, Label: label, Done: done, Total: total}
		h.mu.Lock()
		h.prog = ev
		h.mu.Unlock()
		// Downloads report every chunk; throttle the fan-out, but never
		// swallow a completion event.
		if time.Since(last) < 250*time.Millisecond && done < total {
			return
		}
		last = time.Now()
		h.broadcast(ev)
	}
}

func (h *setupHub) broadcast(v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	for ch := range h.subs {
		select {
		case ch <- b:
		default: // slow subscriber loses an event rather than blocking setup
		}
	}
}

func (h *setupHub) subscribe() chan []byte {
	ch := make(chan []byte, 32)
	h.mu.Lock()
	h.subs[ch] = struct{}{}
	h.mu.Unlock()
	return ch
}

func (h *setupHub) unsubscribe(ch chan []byte) {
	h.mu.Lock()
	delete(h.subs, ch)
	h.mu.Unlock()
}

// newSetupMux serves the three onboarding routes. Token gating is applied by
// the caller (same middleware as the main server).
func newSetupMux(h *setupHub) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/v1/local/state", h.handleState)
	mux.HandleFunc("POST /api/v1/local/bootstrap", h.handleBootstrap)
	mux.HandleFunc("GET /api/v1/local/progress", h.handleProgress)
	return mux
}

func (h *setupHub) handleState(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	resp := map[string]any{
		"phase":       h.phase,
		"hostname":    h.info.Hostname,
		"machine":     map[string]any{"usable_gb": h.info.UsableGB, "uma": h.info.UMA},
		"recommended": h.info.Recommended,
		"models":      h.info.Models,
		"progress":    h.prog,
	}
	if h.errMsg != "" {
		resp["error"] = h.errMsg
	}
	h.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (h *setupHub) handleBootstrap(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ModelID string `json:"model_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}
	id := req.ModelID
	if id == "" {
		id = h.info.Recommended
	}
	var chosen *setupModel
	for i := range h.info.Models {
		if h.info.Models[i].ID == id {
			chosen = &h.info.Models[i]
			break
		}
	}
	if chosen == nil {
		http.Error(w, `{"error":"unknown model"}`, http.StatusBadRequest)
		return
	}
	if !chosen.Fits {
		http.Error(w, `{"error":"model does not fit this machine"}`, http.StatusBadRequest)
		return
	}
	if !h.offerConsent(chosen.ID) {
		http.Error(w, `{"error":"setup already started"}`, http.StatusConflict)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(map[string]string{"model_id": chosen.ID})
}

func (h *setupHub) handleProgress(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	fmt.Fprintf(w, ": connected\n\n")
	flusher.Flush()

	ch := h.subscribe()
	defer h.unsubscribe(ch)
	ticker := time.NewTicker(25 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			fmt.Fprintf(w, ": heartbeat\n\n")
			flusher.Flush()
		case b := <-ch:
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
		}
	}
}

// teeProgress fans one Progress callback out to several sinks (terminal line
// + SSE hub).
func teeProgress(sinks ...localrt.Progress) localrt.Progress {
	return func(stage string, done, total int64, label string) {
		for _, s := range sinks {
			s(stage, done, total, label)
		}
	}
}

// resolveCatalogModel maps a consented model id back to its catalog entry and
// the variant this machine would run.
func resolveCatalogModel(id string) (*localrt.CatalogEntry, *localrt.QuantVariant) {
	budget := localrt.ProbeBudget(true)
	for _, e := range localrt.Catalog() {
		if c := localrt.SelectVariant(e, budget); c != nil && c.Variant.ModelID() == id {
			v := c.Variant
			return e, &v
		}
	}
	return nil, nil
}

// bootstrapMobile is the phone-led first-run: serve the setup API on the
// final port, offer the starter on the terminal too, and block until either
// surface consents. Returns the staged model id, or "" when setup can't
// proceed (nothing fits).
func bootstrapMobile(ctx context.Context, addr, token string) (string, error) {
	hostname, _ := os.Hostname()
	info := buildSetupInfo(hostname)
	anyFits := false
	for _, m := range info.Models {
		if m.Fits {
			anyFits = true
			break
		}
	}
	if !anyFits {
		return "", nil
	}

	hub := newSetupHub(info)
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return "", fmt.Errorf("setup server listen %s: %w", addr, err)
	}
	srv := &http.Server{Handler: tokenMiddleware(token)(newSetupMux(hub))}
	go srv.Serve(ln)
	defer srv.Close()

	// The phone must be able to pair BEFORE consent — it may be the only
	// surface driving this setup — so the QR prints here, while we wait.
	if _, port, err := net.SplitHostPort(addr); err == nil {
		printPairing(os.Stdout, port, token)
	}

	// The terminal path still works: same starter offer, first answer wins.
	go func() {
		entry, choice := localrt.StarterEntry(localrt.Catalog(), localrt.ProbeBudget(true))
		if entry == nil {
			return
		}
		tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
		if err != nil {
			return // headless: the phone is the only consent surface
		}
		defer tty.Close()
		if confirmFirstRun(tty, tty, entry, choice.Variant) {
			if !hub.offerConsent(choice.Variant.ModelID()) {
				fmt.Fprintln(tty, "→ already started from the phone")
			}
		}
	}()

	fmt.Println("no model server found — waiting for setup consent (terminal or phone)…")
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case id := <-hub.consent:
		entry, variant := resolveCatalogModel(id)
		if entry == nil {
			return "", fmt.Errorf("consented model %q no longer resolves against this machine", id)
		}
		hub.setPhase(phaseDownloading)
		prog := teeProgress(progressPrinter(), hub.progressFn())
		if err := installModelWithProgress(ctx, entry, *variant, localrt.DefaultRuntimeTag, prog); err != nil {
			hub.setError(err.Error())
			if errors.Is(err, context.Canceled) {
				return "", err
			}
			return "", fmt.Errorf("phone-led setup: %w", err)
		}
		// Announce the handoff, then free the port for the real server.
		hub.setPhase(phaseStarting)
		time.Sleep(150 * time.Millisecond) // let the SSE event flush to subscribers
		return id, nil
	}
}

// buildSetupInfo prices the catalog against this machine, the same math as
// `proxima models`.
func buildSetupInfo(hostname string) setupInfo {
	budget := localrt.ProbeBudget(true)
	entries := localrt.Catalog()
	staged := map[string]bool{}
	for _, id := range localrt.StagedModelIDs() {
		staged[id] = true
	}
	info := setupInfo{
		Hostname: hostname,
		UsableGB: float64(budget.UsableVRAMBytes) / (1 << 30),
		UMA:      budget.UMA,
	}
	if rec := localrt.RecommendedEntry(entries, budget); rec != nil {
		if c := localrt.SelectVariant(rec.Entry, budget); c != nil {
			info.Recommended = c.Variant.ModelID()
		}
	}
	for _, e := range entries {
		m := setupModel{
			DisplayName: e.DisplayName,
			Description: e.Description,
			Starter:     e.Starter,
		}
		if c := localrt.SelectVariant(e, budget); c != nil {
			m.ID = c.Variant.ModelID()
			m.Fits = true
			m.DownloadGB = float64(e.DownloadBytes(c.Variant)) / 1e9
			m.TokS = localrt.PredictedDecodeTokS(e, c.Variant, budget, !c.ZeroSpill)
			switch c.ReasonKey {
			case "best-large-window":
				m.FitReason = "fits with a large context window"
			case "best-fits":
				m.FitReason = "fits at the 64K floor"
			case "smallest-fits-spilled":
				m.FitReason = "runs spilled to RAM (slow)"
			}
			m.Staged = staged[m.ID]
		} else {
			v := e.Variants[len(e.Variants)-1]
			m.ID = v.ModelID()
			m.DownloadGB = float64(e.DownloadBytes(v)) / 1e9
			m.FitReason = "does not fit this machine"
		}
		info.Models = append(info.Models, m)
	}
	return info
}
