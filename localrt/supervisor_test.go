package localrt

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestMain doubles as the fake llama-server router: when GO_HELPER_ROUTER is
// set, the test binary serves the router API instead of running tests.
func TestMain(m *testing.M) {
	if os.Getenv("GO_HELPER_ROUTER") == "1" {
		runFakeRouter()
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// runFakeRouter serves just enough of llama-server's router API for the
// supervisor: /health, /models, /models/load, /v1/chat/completions, /slots,
// /metrics.
func runFakeRouter() {
	port := ""
	for i, a := range os.Args {
		if a == "--port" && i+1 < len(os.Args) {
			port = os.Args[i+1]
		}
	}
	var mu sync.Mutex
	status := map[string]string{"tiny-model": "unloaded"}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		var data []map[string]any
		for id, s := range status {
			data = append(data, map[string]any{"id": id, "status": map[string]string{"value": s}})
		}
		json.NewEncoder(w).Encode(map[string]any{"data": data})
	})
	mux.HandleFunc("/models/load", func(w http.ResponseWriter, r *http.Request) {
		var body struct{ Model string }
		json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		status[body.Model] = "loaded"
		mu.Unlock()
		w.Write([]byte("{}"))
	})
	mux.HandleFunc("/models/unload", func(w http.ResponseWriter, r *http.Request) {
		var body struct{ Model string }
		json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		status[body.Model] = "unloaded"
		mu.Unlock()
		w.Write([]byte("{}"))
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{
				"reasoning_content": "The user asks for the capital of France.",
				"content":           "Paris",
			}}},
		})
	})
	mux.HandleFunc("/slots", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`[{"is_processing": false}]`))
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("llamacpp:requests_processing 0\n"))
	})
	http.ListenAndServe("127.0.0.1:"+port, mux)
}

// helperExe writes a wrapper script that re-execs this test binary as the
// fake router.
func helperExe(t *testing.T) string {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	script := filepath.Join(t.TempDir(), "llama-server")
	body := "#!/bin/sh\nGO_HELPER_ROUTER=1 exec \"" + self + "\" \"$@\"\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return script
}

func TestSupervisorLifecycle(t *testing.T) {
	if testing.Short() {
		t.Skip("spawns processes")
	}
	t.Setenv("VEGA_HOME", t.TempDir())
	modelsDir := ModelsDir()
	os.MkdirAll(modelsDir, 0o755)

	sup := NewSupervisor(t.TempDir(), modelsDir)
	sup.Exe = helperExe(t)
	sup.Port = freePort()
	if err := sup.Start(20 * time.Second); err != nil {
		t.Fatal(err)
	}
	defer sup.Stop()

	// State file describes the running server.
	state := ReadServerState()
	if state == nil || state.BaseURL != sup.BaseURL() || state.APIKey != sup.APIKey || state.PID == 0 {
		t.Fatalf("state: %+v", state)
	}

	// Model management round-trip.
	models, err := sup.Models()
	if err != nil || models["tiny-model"] != "unloaded" {
		t.Fatalf("models: %v err=%v", models, err)
	}
	ready, err := sup.EnsureModelReady("tiny-model", 30*time.Second)
	if err != nil || !ready {
		t.Fatalf("ensure ready: %v err=%v", ready, err)
	}
	if _, err := sup.EnsureModelReady("missing-model", time.Second); err == nil {
		t.Error("missing model should error")
	}
	if !sup.IsIdle("tiny-model") || !sup.IsIdle("") {
		t.Error("fake router should read idle")
	}

	// Crash the router; the watchdog restarts it (1s backoff) and the
	// primary model is re-readied.
	sup.SetPrimaryModel("tiny-model")
	sup.mu.Lock()
	oldPID := sup.cmd.Process.Pid
	sup.cmd.Process.Kill()
	sup.mu.Unlock()

	deadline := time.Now().Add(15 * time.Second)
	restarted := false
	for time.Now().Before(deadline) {
		sup.mu.Lock()
		pid := 0
		if sup.cmd != nil && sup.cmd.Process != nil {
			pid = sup.cmd.Process.Pid
		}
		restarts := sup.restarts
		sup.mu.Unlock()
		if restarts > 0 && pid != 0 && pid != oldPID {
			if _, err := sup.Models(); err == nil {
				restarted = true
				break
			}
		}
		time.Sleep(300 * time.Millisecond)
	}
	if !restarted {
		t.Fatal("watchdog did not restart the router")
	}

	// Stop removes the state file and does not restart again.
	sup.Stop()
	time.Sleep(500 * time.Millisecond)
	if ReadServerState() != nil {
		t.Error("state file survived Stop")
	}
}

func TestStableAPIKeyPersists(t *testing.T) {
	t.Setenv("VEGA_HOME", t.TempDir())
	k1 := stableAPIKey()
	k2 := stableAPIKey()
	if k1 != k2 || len(k1) < 16 {
		t.Errorf("keys: %q vs %q", k1, k2)
	}
}

func TestDownloadModelResumes(t *testing.T) {
	t.Setenv("VEGA_HOME", t.TempDir())
	payload := strings.Repeat("W", 4096)
	var ranged bool
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rng := r.Header.Get("Range"); rng != "" {
			ranged = true
			var from int
			fmt.Sscanf(rng, "bytes=%d-", &from)
			w.Header().Set("Content-Length", fmt.Sprint(len(payload)-from))
			w.WriteHeader(http.StatusPartialContent)
			w.Write([]byte(payload[from:]))
			return
		}
		w.Write([]byte(payload))
	}))
	defer ts.Close()
	oldURL := hfResolveURL
	hfResolveURL = ts.URL + "/%s/%s"
	defer func() { hfResolveURL = oldURL }()

	entry := &CatalogEntry{ID: "t", Repo: "test/t",
		Variants: []QuantVariant{{Quant: "Q4",
			Files: []AssetFile{{Path: "t-Q4.gguf", SizeBytes: int64(len(payload))}}}},
		NCtxTrain: 65536, FullLayers: 1, PerLayerF16: 64,
		MMProj: &AssetFile{Path: "mmproj-BF16.gguf", SizeBytes: 8, Local: "mmproj-t-BF16.gguf"},
	}
	os.MkdirAll(ModelsDir(), 0o755)

	// Seed a partial download; the fetch must resume, not restart.
	os.WriteFile(filepath.Join(ModelsDir(), "t-Q4.gguf.part"), []byte(payload[:1000]), 0o644)

	if err := DownloadModel(context.Background(), entry, entry.Variants[0], nil); err != nil {
		t.Fatal(err)
	}
	if !ranged {
		t.Error("partial download did not resume with a Range request")
	}
	got, err := os.ReadFile(filepath.Join(ModelsDir(), "t-Q4.gguf"))
	if err != nil || string(got) != payload {
		t.Fatalf("weights content wrong (len %d)", len(got))
	}
	// Companion asset lands in assets/, not the models dir.
	if _, err := os.Stat(filepath.Join(AssetsDir(), "mmproj-t-BF16.gguf")); err != nil {
		t.Error("mmproj not in assets dir")
	}
	if got := StagedModelIDs(); len(got) != 1 || got[0] != "t-Q4" {
		t.Errorf("staged ids: %v", got)
	}

	// Re-download of a complete file is a no-op.
	if err := DownloadModel(context.Background(), entry, entry.Variants[0], nil); err != nil {
		t.Fatal(err)
	}
}
