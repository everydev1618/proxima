package localrt

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestMaybeGrowWindow runs the full growth execution: a staged model at the
// 64K rung overflows, the ladder grants 96K, the router bounces with a
// regenerated preset restoring the grown window, and readiness is proven.
func TestMaybeGrowWindow(t *testing.T) {
	if testing.Short() {
		t.Skip("spawns processes")
	}
	t.Setenv("VEGA_HOME", t.TempDir())
	os.MkdirAll(ModelsDir(), 0o755)
	writeTestGGUF(t, ModelsDir(), "tiny-Q4_K_M.gguf") // native 262144

	// The fake router serves "tiny-model"; growth only needs it healthy and
	// idle, plus a working chat endpoint for the touch. Register the staged
	// id by pointing the touch at whatever model is asked for.
	sup := NewSupervisor(t.TempDir(), ModelsDir())
	sup.Exe = helperExe(t)
	sup.Port = freePort()
	// Tight card, roomy host: the launch decision holds the 64K floor
	// (runtime overhead alone exceeds usable VRAM), and the next rung fits
	// VRAM+RAM so growth can grant it.
	budget := HardwareBudget{UsableVRAMBytes: 100 << 20, RAMAvailableBytes: 32 * gib}
	presetPath := filepath.Join(RuntimesRoot(), "presets.ini")
	sup.PresetPath = presetPath
	sup.PreSpawn = func() error {
		_, err := GeneratePresets(ModelsDir(), budget, presetPath, nil)
		return err
	}
	if err := sup.Start(20 * time.Second); err != nil {
		t.Fatal(err)
	}
	defer sup.Stop()

	// The launch decision on this budget is the floor rung.
	if d := ReadPresetDecisions(presetPath)["tiny-Q4_K_M"]; d == nil || d.Window != 65536 {
		t.Fatalf("launch preset: %+v", d)
	}

	window, ok := MaybeGrowWindow(sup, "tiny-Q4_K_M", budget)
	if !ok || window != 98304 {
		t.Fatalf("grow: window=%d ok=%v", window, ok)
	}
	if got := LoadWindowOverrides()["tiny-Q4_K_M"]; got != 98304 {
		t.Errorf("override not persisted: %d", got)
	}
	// The bounce regenerated presets; the grown window is restored there.
	if d := ReadPresetDecisions(presetPath)["tiny-Q4_K_M"]; d == nil || d.Window != 98304 {
		t.Errorf("preset after bounce: %+v", d)
	}
	// The router survived the bounce.
	if _, err := sup.Models(); err != nil {
		t.Errorf("router unhealthy after bounce: %v", err)
	}

	// A model with no preset record declines.
	if _, ok := MaybeGrowWindow(sup, "unknown-model", budget); ok {
		t.Error("unknown model must not grow")
	}
}
