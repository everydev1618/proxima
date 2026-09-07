// Persisted context-growth state for the managed runtime. Port of
// hermes-agent's local_runtime/growth.py (MIT, Nous Research — see NOTICE).
//
// Scope guard: only a server THIS process supervises grows. Detected external
// servers keep their own policies.

package localrt

import (
	"encoding/json"
	"os"
	"path/filepath"
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
