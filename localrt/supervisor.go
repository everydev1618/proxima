// Supervision of one llama-server in router mode. Port of hermes-agent's
// local_runtime/supervisor.py (MIT, Nous Research — see NOTICE).
//
// The router process is ours (restart with backoff on crash); router children
// are its problem — child failures surface via GET /models, never
// auto-retried here. Learned on real hardware: health-200 is NOT readiness —
// every readiness claim requires a touch generation (temp-0, expected token,
// generous budget, reasoning_content scanned); always dial 127.0.0.1 —
// resolving localhost adds ~2s per request on some platforms via IPv6
// fallback.

package localrt

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// TouchPrompt/TouchExpect: the readiness proof.
	TouchPrompt = "Reply with exactly one word: the capital of France."
	TouchExpect = "paris"

	// Chosen once and reused across restarts: sessions persist the resolved
	// base_url, so an ephemeral port would strand every resumed session
	// after each restart. Deliberately NOT 8080 (a user's own llama-server /
	// Ollama-adjacent stack) and not hermes-agent's 18434 (both harnesses on
	// one box must not fight over a port).
	defaultRouterPort = 18535

	// IdleUnloadAfter: a model that has gone quiet gets its VRAM back after
	// this long. A constant, not a knob: long enough that an active
	// conversation never trips it, short enough that a wandered-off session
	// frees ~20 GiB within the hour. No exemptions: demand reloads anything
	// the user comes back to.
	IdleUnloadAfter = 15 * time.Minute
)

var restartBackoff = []time.Duration{1 * time.Second, 5 * time.Second, 15 * time.Second, 60 * time.Second}

var residentStatuses = map[string]bool{"loaded": true, "ready": true}

// StatePath is the endpoint state for other processes (provider resolution
// routes managed-server requests from this).
func StatePath() string { return filepath.Join(RuntimesRoot(), "server.json") }

// ServerState is the persisted endpoint identity of the managed server.
type ServerState struct {
	BaseURL string `json:"base_url"`
	APIKey  string `json:"api_key"`
	PID     int    `json:"pid"`
}

// ReadServerState returns the persisted state, or nil.
func ReadServerState() *ServerState {
	raw, err := os.ReadFile(StatePath())
	if err != nil {
		return nil
	}
	var s ServerState
	if json.Unmarshal(raw, &s) != nil || s.BaseURL == "" {
		return nil
	}
	return &s
}

func freePort() int {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// stablePort returns the stable default port, or an ephemeral one only when
// something else already listens there.
func stablePort() int {
	l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", defaultRouterPort))
	if err == nil {
		l.Close()
		return defaultRouterPort
	}
	slog.Warn("managed llama-server port busy; falling back to an ephemeral port — existing sessions may need a model re-pick",
		"port", defaultRouterPort)
	return freePort()
}

// stableAPIKey returns one key for the life of the install, persisted beside
// the runtimes. Endpoint identity must survive restarts as a UNIT — sessions
// persist base_url + api_key, so a per-boot key strands every resumed
// session on HTTP 401 exactly as a per-boot port would on connection errors.
func stableAPIKey() string {
	keyPath := filepath.Join(RuntimesRoot(), ".api_key")
	if raw, err := os.ReadFile(keyPath); err == nil {
		if key := strings.TrimSpace(string(raw)); len(key) >= 16 {
			return key
		}
	}
	buf := make([]byte, 24)
	rand.Read(buf)
	key := base64.RawURLEncoding.EncodeToString(buf)
	if err := os.MkdirAll(filepath.Dir(keyPath), 0o755); err == nil {
		if err := os.WriteFile(keyPath, []byte(key), 0o600); err != nil {
			slog.Warn("could not persist api key; sessions will need a re-pick after restart", "err", err)
		}
	}
	return key
}

// Supervisor owns one llama-server router process.
type Supervisor struct {
	InstallDir string
	ModelsDir  string
	ModelsMax  int
	Port       int
	APIKey     string
	ExtraArgs  []string
	LogPath    string
	PresetPath string
	// Exe overrides the server binary (tests); empty resolves from InstallDir.
	Exe string
	// PreSpawn, when set, runs before every spawn (first boot, crash
	// restarts, bounces) — the place to regenerate launch presets so a
	// respawned router always starts with current policy.
	PreSpawn func() error

	client *http.Client

	mu           sync.Mutex
	cmd          *exec.Cmd
	logFile      *os.File
	primaryModel string
	restarts     int
	stopping     bool
	gen          int // watchdog generation; a stale watchdog must not respawn
	idleSince    map[string]time.Time
}

// NewSupervisor fills defaults: stable port, persisted API key, log beside
// the models dir.
func NewSupervisor(installDir, modelsDir string) *Supervisor {
	return &Supervisor{
		InstallDir: installDir,
		ModelsDir:  modelsDir,
		ModelsMax:  4,
		Port:       stablePort(),
		APIKey:     stableAPIKey(),
		LogPath:    filepath.Join(filepath.Dir(modelsDir), "logs", "llama-server.log"),
		client:     &http.Client{},
		idleSince:  map[string]time.Time{},
	}
}

// ── endpoints ────────────────────────────────────────────────

// BaseURL is the OpenAI-compatible API root of the managed server.
func (s *Supervisor) BaseURL() string { return fmt.Sprintf("http://127.0.0.1:%d/v1", s.Port) }

func (s *Supervisor) url(route string) string {
	return fmt.Sprintf("http://127.0.0.1:%d%s", s.Port, route)
}

func (s *Supervisor) httpClient() *http.Client {
	if s.client == nil {
		s.client = &http.Client{}
	}
	return s.client
}

func (s *Supervisor) request(route string, body any, timeout time.Duration, out any) error {
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reader = bytes.NewReader(raw)
	}
	method := http.MethodGet
	if body != nil {
		method = http.MethodPost
	}
	req, err := http.NewRequest(method, s.url(route), reader)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.APIKey)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	client := *s.httpClient()
	client.Timeout = timeout
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode >= 400 {
		return fmt.Errorf("%s %s: %s: %.200s", method, route, resp.Status, raw)
	}
	if out != nil && len(raw) > 0 {
		return json.Unmarshal(raw, out)
	}
	return nil
}

// ── lifecycle ────────────────────────────────────────────────

func (s *Supervisor) exePath() (string, error) {
	if s.Exe != "" {
		return s.Exe, nil
	}
	return ServerBinary(s.InstallDir)
}

func (s *Supervisor) spawnLocked() error {
	if s.PreSpawn != nil {
		if err := s.PreSpawn(); err != nil {
			slog.Warn("pre-spawn hook failed; spawning with previous state", "err", err)
		}
	}
	exe, err := s.exePath()
	if err != nil {
		return err
	}
	args := []string{
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(s.Port),
		"--api-key", s.APIKey,
		"--models-dir", s.ModelsDir,
		"--models-max", strconv.Itoa(s.ModelsMax),
		// Residency contract: a chat request to a staged-but-unloaded model
		// loads it (slow first token) instead of a bare 400/404 after an eject.
		"--models-autoload",
		"--metrics", // supervisor telemetry needs it
		"--slots",   // /slots endpoint is opt-in; IsIdle reads it
		"--no-webui",
		"--jinja",
		// Direct I/O on model load bypasses the page cache so a multi-GB
		// load doesn't evict half the OS cache.
		"-dio",
	}
	if s.PresetPath != "" {
		if _, err := os.Stat(s.PresetPath); err == nil {
			args = append(args, "--models-preset", s.PresetPath)
		}
	}
	args = append(args, s.ExtraArgs...)

	if err := os.MkdirAll(filepath.Dir(s.LogPath), 0o755); err != nil {
		return err
	}
	if s.logFile != nil {
		// The crash-restart loop respawns repeatedly; each restart would
		// otherwise leak one fd.
		s.logFile.Close()
	}
	logFile, err := os.OpenFile(s.LogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	s.logFile = logFile
	fmt.Fprintf(logFile, "\n# spawn: %s %s\n", exe, strings.Join(args, " "))

	cmd := exec.Command(exe, args...)
	cmd.Dir = filepath.Dir(exe)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	setProcessGroup(cmd) // children die with the router on our kill
	if err := cmd.Start(); err != nil {
		return err
	}
	s.cmd = cmd
	slog.Info("llama-server router spawned", "pid", cmd.Process.Pid, "port", s.Port)
	// State goes down at SPAWN, not after health: endpoint resolution treats
	// a live-pid-but-not-yet-healthy server as "starting" rather than
	// "unconfigured", so a readiness probe racing the boot doesn't throw the
	// app back to onboarding.
	s.writeState()
	return nil
}

func (s *Supervisor) writeState() {
	pid := 0
	if s.cmd != nil && s.cmd.Process != nil {
		pid = s.cmd.Process.Pid
	}
	raw, _ := json.Marshal(ServerState{BaseURL: s.BaseURL(), APIKey: s.APIKey, PID: pid})
	os.MkdirAll(filepath.Dir(StatePath()), 0o755)
	os.WriteFile(StatePath(), raw, 0o600)
}

// Start spawns the router, waits for health, and begins the crash watchdog.
func (s *Supervisor) Start(healthTimeout time.Duration) error {
	s.mu.Lock()
	s.stopping = false
	s.gen++
	gen := s.gen
	if err := s.spawnLocked(); err != nil {
		s.mu.Unlock()
		return err
	}
	s.mu.Unlock()
	if err := s.waitHealth(healthTimeout); err != nil {
		return err
	}
	s.mu.Lock()
	s.writeState()
	s.mu.Unlock()
	go s.watch(gen)
	return nil
}

// Bounce restarts the router synchronously (kill, respawn via Start, wait
// healthy). Used when the launch policy or staged set changed — the router's
// model list and windows are spawn-only. Sessions ride through on the stable
// port + persisted key; the watchdog generation guard keeps the outgoing
// watchdog from double-respawning.
func (s *Supervisor) Bounce(healthTimeout time.Duration) error {
	s.mu.Lock()
	s.stopping = true
	cmd := s.cmd
	s.mu.Unlock()
	if cmd != nil && cmd.Process != nil {
		killProcessGroup(cmd, 10*time.Second)
	}
	return s.Start(healthTimeout)
}

func (s *Supervisor) procExited() (int, bool) {
	s.mu.Lock()
	cmd := s.cmd
	s.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		return 0, true
	}
	if cmd.ProcessState != nil {
		return cmd.ProcessState.ExitCode(), true
	}
	return 0, false
}

func (s *Supervisor) waitHealth(timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if rc, exited := s.procExited(); exited {
			return fmt.Errorf("llama-server exited rc=%d during startup (log: %s)", rc, s.LogPath)
		}
		req, _ := http.NewRequest(http.MethodGet, s.url("/health"), nil)
		client := *s.httpClient()
		client.Timeout = 3 * time.Second
		if resp, err := client.Do(req); err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("llama-server not healthy after %s (log: %s)", timeout, s.LogPath)
}

// watch restarts the router (not its children) on crash, with backoff. gen
// guards against a superseded watchdog (Bounce/Start started a newer one)
// respawning on top of the new generation's process.
func (s *Supervisor) watch(gen int) {
	for {
		s.mu.Lock()
		stale := s.stopping || s.gen != gen
		cmd := s.cmd
		s.mu.Unlock()
		if stale || cmd == nil {
			return
		}
		err := cmd.Wait() // reaps; returns when the process exits
		s.mu.Lock()
		stale = s.stopping || s.gen != gen
		restarts := s.restarts
		s.mu.Unlock()
		if stale {
			return
		}
		backoff := restartBackoff[min(restarts, len(restartBackoff)-1)]
		slog.Warn("llama-server exited; restarting", "err", err, "attempt", restarts+1, "backoff", backoff)
		time.Sleep(backoff)

		s.mu.Lock()
		if s.stopping || s.gen != gen {
			s.mu.Unlock()
			return
		}
		s.restarts++
		// Kill model children orphaned by the router crash before respawn —
		// each holds gigabytes of VRAM that must come back before the new
		// router loads models next to the ghosts. Process-group kill reaches
		// them even though their parent is gone.
		killProcessGroup(cmd, 0)
		spawnErr := s.spawnLocked()
		primary := s.primaryModel
		s.mu.Unlock()
		if spawnErr != nil {
			slog.Error("llama-server restart failed", "err", spawnErr)
			return
		}
		if err := s.waitHealth(120 * time.Second); err != nil {
			slog.Error("llama-server restart failed", "err", err)
			continue
		}
		if primary != "" {
			if _, err := s.EnsureModelReady(primary, 600*time.Second); err != nil {
				slog.Warn("primary model not ready after restart", "model", primary, "err", err)
			}
		}
	}
}

// Stop terminates the router AND its model children (each child holds
// gigabytes of VRAM; killing only the router orphans them with the weights
// still resident).
func (s *Supervisor) Stop() {
	s.mu.Lock()
	s.stopping = true
	cmd := s.cmd
	logFile := s.logFile
	s.logFile = nil
	s.mu.Unlock()
	os.Remove(StatePath())
	if cmd != nil && cmd.Process != nil {
		killProcessGroup(cmd, 15*time.Second)
	}
	if logFile != nil {
		logFile.Close()
	}
}

// SetPrimaryModel declares the model the watchdog re-readies after a crash
// restart. The declaration is durable; an eject is not.
func (s *Supervisor) SetPrimaryModel(modelID string) {
	s.mu.Lock()
	s.primaryModel = modelID
	s.mu.Unlock()
}

// ── model management (router endpoints) ──────────────────────

// Models returns {model_id: status_value} from GET /models.
func (s *Supervisor) Models() (map[string]string, error) {
	var out struct {
		Data []struct {
			ID     string `json:"id"`
			Status struct {
				Value string `json:"value"`
			} `json:"status"`
		} `json:"data"`
	}
	if err := s.request("/models", nil, 30*time.Second, &out); err != nil {
		return nil, err
	}
	models := map[string]string{}
	for _, m := range out.Data {
		status := m.Status.Value
		if status == "" {
			status = "unknown"
		}
		models[m.ID] = status
	}
	return models, nil
}

// LoadModel asks the router to load a model.
func (s *Supervisor) LoadModel(modelID string, timeout time.Duration) error {
	return s.request("/models/load", map[string]string{"model": modelID}, timeout, nil)
}

// UnloadModel frees the child's VRAM now. Momentary: never touches the
// primary-model declaration — the declaration is durable, an eject is not.
func (s *Supervisor) UnloadModel(modelID string) error {
	if err := s.request("/models/unload", map[string]string{"model": modelID}, 120*time.Second, nil); err != nil {
		return err
	}
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		models, err := s.Models()
		if err != nil {
			return nil
		}
		status := models[modelID]
		if !residentStatuses[status] && status != "unloading" {
			return nil
		}
		time.Sleep(300 * time.Millisecond)
	}
	return nil
}

// SweepIdle unloads models idle past IdleUnloadAfter; returns their ids.
// Idle = no busy slots and no queued work, tracked per model across calls; a
// model seen busy resets its clock.
func (s *Supervisor) SweepIdle() []string {
	now := time.Now()
	var unloaded []string
	statuses, err := s.Models()
	if err != nil {
		return unloaded
	}
	s.mu.Lock()
	if s.idleSince == nil {
		s.idleSince = map[string]time.Time{}
	}
	s.mu.Unlock()
	for modelID, status := range statuses {
		if !residentStatuses[status] || !s.IsIdle(modelID) {
			s.mu.Lock()
			delete(s.idleSince, modelID)
			s.mu.Unlock()
			continue
		}
		s.mu.Lock()
		firstIdle, seen := s.idleSince[modelID]
		if !seen {
			firstIdle = now
			s.idleSince[modelID] = now
		}
		s.mu.Unlock()
		if now.Sub(firstIdle) < IdleUnloadAfter {
			continue
		}
		if err := s.UnloadModel(modelID); err != nil {
			slog.Warn("idle unload failed", "model", modelID, "err", err)
			continue
		}
		s.mu.Lock()
		delete(s.idleSince, modelID)
		s.mu.Unlock()
		unloaded = append(unloaded, modelID)
		slog.Info("idle-unloaded model", "model", modelID, "idle", now.Sub(firstIdle))
	}
	return unloaded
}

// TouchGenerate is the readiness proof. Generous budget + reasoning_content
// scan — small token budgets false-fail reasoning models, which spend their
// first tokens thinking.
func (s *Supervisor) TouchGenerate(modelID string, timeout time.Duration) bool {
	var resp struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
	}
	err := s.request("/v1/chat/completions", map[string]any{
		"model":      modelID,
		"messages":   []map[string]string{{"role": "user", "content": TouchPrompt}},
		"max_tokens": 512, "temperature": 0,
	}, timeout, &resp)
	if err != nil || len(resp.Choices) == 0 {
		slog.Warn("touch generation failed", "model", modelID, "err", err)
		return false
	}
	blob := strings.ToLower(resp.Choices[0].Message.Content + " " + resp.Choices[0].Message.ReasoningContent)
	return strings.Contains(blob, TouchExpect)
}

// EnsureModelReady loads if needed, then proves readiness with a touch
// generation.
func (s *Supervisor) EnsureModelReady(modelID string, timeout time.Duration) (bool, error) {
	models, err := s.Models()
	if err != nil {
		return false, err
	}
	status, ok := models[modelID]
	if !ok {
		return false, fmt.Errorf("model %s not present in models dir", modelID)
	}
	if !residentStatuses[status] {
		if err := s.LoadModel(modelID, timeout); err != nil {
			return false, err
		}
	}
	return s.TouchGenerate(modelID, 300*time.Second), nil
}

// ── telemetry ────────────────────────────────────────────────

// IsIdle reports no processing requests and no busy slots. Router quirk:
// /slots and /metrics are per-child and require ?model= (bare calls 400).
// With modelID checks that one child; with "" every loaded child.
func (s *Supervisor) IsIdle(modelID string) bool {
	var loaded []string
	if modelID != "" {
		loaded = []string{modelID}
	} else {
		models, err := s.Models()
		if err != nil {
			return false
		}
		for m, status := range models {
			if residentStatuses[status] {
				loaded = append(loaded, m)
			}
		}
	}
	for _, mid := range loaded {
		var slots []struct {
			IsProcessing bool `json:"is_processing"`
		}
		if err := s.request("/slots?model="+mid, nil, 10*time.Second, &slots); err != nil {
			return false
		}
		for _, slot := range slots {
			if slot.IsProcessing {
				return false
			}
		}
		req, _ := http.NewRequest(http.MethodGet, s.url("/metrics?model="+mid), nil)
		req.Header.Set("Authorization", "Bearer "+s.APIKey)
		client := *s.httpClient()
		client.Timeout = 10 * time.Second
		resp, err := client.Do(req)
		if err != nil {
			return false
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		for _, line := range strings.Split(string(raw), "\n") {
			if strings.HasPrefix(line, "llamacpp:requests_processing") {
				fields := strings.Fields(line)
				if v, err := strconv.ParseFloat(fields[len(fields)-1], 64); err == nil && v != 0 {
					return false
				}
			}
		}
	}
	return true
}

// StartIdleSweeper runs the idle-residency loop: every couple of minutes,
// unload models idle past the threshold. Exits when the supervisor stops.
func (s *Supervisor) StartIdleSweeper() {
	go func() {
		for {
			time.Sleep(2 * time.Minute)
			s.mu.Lock()
			stopping := s.stopping
			s.mu.Unlock()
			if stopping {
				return
			}
			s.SweepIdle()
		}
	}()
}
