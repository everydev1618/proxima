// Persisted context-growth state for the managed runtime. Port of
// hermes-agent's local_runtime/growth.py (MIT, Nous Research — see NOTICE).
//
// Scope guard: only a server THIS process supervises grows. Detected external
// servers keep their own policies.

package localrt

import (
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func windowOverridesPath() string {
	return filepath.Join(RuntimesRoot(), "window_overrides.json")
}

// LoadWindowOverrides returns model_id -> granted window. Empty on any read
// problem.
func LoadWindowOverrides() map[string]int {
	out := map[string]int{}
	raw, err := os.ReadFile(windowOverridesPath())
	if err != nil {
		return out
	}
	_ = json.Unmarshal(raw, &out)
	return out
}

func writeOverrides(overrides map[string]int) error {
	path := windowOverridesPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(overrides, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

// SaveWindowOverride records a grown window for a model.
func SaveWindowOverride(modelID string, window int) error {
	overrides := LoadWindowOverrides()
	overrides[modelID] = window
	return writeOverrides(overrides)
}

// ClearWindowOverride drops a model's growth state (delete/re-download paths).
func ClearWindowOverride(modelID string) error {
	overrides := LoadWindowOverrides()
	if _, ok := overrides[modelID]; !ok {
		return nil
	}
	delete(overrides, modelID)
	return writeOverrides(overrides)
}

// MaybeGrowWindow runs one growth evaluation + execution for a managed
// model, called from govega's context-pressure hook when a request
// overflowed the window. Occupancy is confirmed by construction — the server
// itself refused the context. On a grant: persist the override, bounce the
// router (PreSpawn regenerates presets, restoring the grown window), and
// prove readiness. Returns (newWindow, true) only when the retry can
// actually succeed.
//
// Budget is the CAPACITY budget (ProbeBudget(true)), not live-free: growth
// executes via a server bounce, so the grown instance loads onto a freed
// card — live-free would read the model's own residency as unavailable and
// veto rungs that fit.
func MaybeGrowWindow(sup *Supervisor, modelID string, budget HardwareBudget) (int, bool) {
	presetPath := filepath.Join(RuntimesRoot(), "presets.ini")
	cur := ReadPresetDecisions(presetPath)[modelID]
	if cur == nil || cur.Window <= 0 {
		return 0, false
	}

	var gguf string
	for _, p := range StagedModels() {
		if ModelIDFromStem(strings.TrimSuffix(filepath.Base(p), ".gguf")) == modelID {
			gguf = p
			break
		}
	}
	if gguf == "" {
		return 0, false
	}
	header, err := ReadGGUFHeader(gguf)
	if err != nil {
		slog.Debug("growth skip: unreadable gguf", "model", modelID, "err", err)
		return 0, false
	}
	profile := ProfileFromGGUF(header)
	if entry, _ := FindEntryForModel(Catalog(), modelID); entry != nil && entry.MTP {
		profile.KVScale = 1.2
	}

	decision := EvaluateGrowth(profile, budget, GrowthInputs{
		CurrentWindow:      cur.Window,
		ServerIdle:         sup.IsIdle(modelID),
		FlashAttention:     true,
		OccupancyConfirmed: true, // the server refused the context; the edge fired
	})
	if decision.Action != GrowthGrow || decision.NextWindow == 0 {
		slog.Debug("growth declined", "model", modelID, "action", decision.Action, "reason", decision.Reason)
		return 0, false
	}

	slog.Info("context growth", "model", modelID, "reason", decision.Reason)
	if err := SaveWindowOverride(modelID, decision.NextWindow); err != nil {
		return 0, false
	}
	if err := sup.Bounce(2 * time.Minute); err != nil {
		// The override still lands at the next boot; report no growth NOW so
		// the caller compresses instead of overflowing a stale window.
		slog.Warn("growth bounce failed; compression proceeds", "model", modelID, "err", err)
		return 0, false
	}
	ready, err := sup.EnsureModelReady(modelID, 600*time.Second)
	if err != nil || !ready {
		slog.Warn("grown window not ready", "model", modelID, "err", err)
		return 0, false
	}
	return decision.NextWindow, true
}
