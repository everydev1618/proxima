// Bootstrap for the managed runtime: installed binaries -> running supervised
// server. Port (simplified) of hermes-agent's local_runtime/bootstrap.py
// (MIT, Nous Research — see NOTICE).
//
// Divergence from the original: no cross-process adoption of an incumbent
// server. lite-vega is typically the only supervisor on the machine, so a
// live state-file server is stopped and replaced by a fresh boot with
// regenerated presets — sessions ride through on the stable port + persisted
// key exactly as they do across a supervised restart.

package localrt

import (
	"fmt"
	"log/slog"
	"os"
	"syscall"
	"time"
)

// EnsureManagedRuntime boots the managed llama-server over the staged models
// and returns its supervisor. Preconditions: at least one staged model and
// one installed runtime tag (callers surface `lite-vega pull` guidance
// otherwise).
func EnsureManagedRuntime(tag string) (*Supervisor, error) {
	staged := StagedModels()
	if len(staged) == 0 {
		return nil, fmt.Errorf("no models staged in %s", ModelsDir())
	}

	backend := SelectBackend(DetectGPUVendor(), "")
	// Boot ladder: serve what is INSTALLED, never download here — a
	// multi-minute inline download does not belong on the boot path. The
	// requested tag is preferred; otherwise the newest installed tag serves.
	if tag == "" {
		tag = DefaultRuntimeTag
	}
	installed := InstalledTags()
	if !contains(installed, tag) {
		if len(installed) == 0 {
			return nil, fmt.Errorf("no llama.cpp build installed; run `lite-vega pull` first")
		}
		slog.Info("configured runtime tag not installed; serving newest installed",
			"configured", tag, "serving", installed[0])
		tag = installed[0]
	}
	plan, err := ResolveAssets(tag, backend, "", "")
	if err != nil {
		return nil, err
	}
	installDir := plan.InstallDir()
	if !ManifestVerified(installDir + "/manifest.json") {
		return nil, fmt.Errorf("runtime %s/%s not verified; run `lite-vega pull` to (re)install", tag, backend)
	}

	// A previous supervisor (this or another process) may still hold the
	// port; take over cleanly.
	stopStateServer()

	// Launch policy per staged model, priced against CAPACITY, not live free
	// VRAM: this may run while an outgoing server still holds the card, and
	// its memory is freed before the new instance loads anything.
	presetPath := RuntimesRoot() + "/presets.ini"
	entries, err := GeneratePresets(ModelsDir(), ProbeBudget(true), presetPath, Catalog())
	if err != nil {
		// Degradation ladder: a STALE policy still beats no policy. Keep
		// serving with the previous INI when one exists.
		if _, statErr := os.Stat(presetPath); statErr != nil {
			presetPath = ""
		}
		slog.Error("preset generation failed; serving with previous/no launch policy", "err", err)
	}
	for _, e := range entries {
		if e.Refusal != "" {
			slog.Warn("model refused by physics check", "detail", e.Refusal)
		}
	}

	sup := NewSupervisor(installDir, ModelsDir())
	sup.PresetPath = presetPath
	if err := sup.Start(120 * time.Second); err != nil {
		// start can fail after the router process exists (health timeout):
		// leaving it running unsupervised strands its VRAM behind a port
		// nothing will clean up.
		sup.Stop()
		return nil, err
	}
	sup.StartIdleSweeper()
	slog.Info("managed llama-server up", "base_url", sup.BaseURL(), "backend", backend, "tag", tag)
	return sup, nil
}

// stopStateServer best-effort stops the server the state file points at. The
// state pid is ours by contract — the file only ever describes the managed
// server.
func stopStateServer() {
	state := ReadServerState()
	if state == nil || state.PID <= 0 {
		return
	}
	proc, err := os.FindProcess(state.PID)
	if err != nil {
		return
	}
	if proc.Signal(syscall.Signal(0)) != nil {
		os.Remove(StatePath()) // stale file, dead pid
		return
	}
	slog.Info("stopping incumbent managed llama-server", "pid", state.PID)
	proc.Signal(syscall.SIGTERM)
	for i := 0; i < 50; i++ {
		if proc.Signal(syscall.Signal(0)) != nil {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	proc.Kill()
}

func contains(list []string, v string) bool {
	for _, x := range list {
		if x == v {
			return true
		}
	}
	return false
}
