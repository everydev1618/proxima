package localrt

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTestGGUF writes a small dense GGUF into dir and returns its path.
func writeTestGGUF(t *testing.T, dir, name string) string {
	t.Helper()
	b := &ggufBuilder{}
	b.str("general.architecture", "qwen3").
		u32("qwen3.block_count", 2).
		u32("qwen3.context_length", 262144).
		u32("qwen3.embedding_length", 512).
		u32("qwen3.attention.head_count", 4).
		u32("qwen3.attention.head_count_kv", 2).
		f32("general.sampling.temp", 0.7).
		tensor("token_embd.weight", []uint64{512, 100}, 1)
	src := b.write(t)
	dest := filepath.Join(dir, name)
	raw, _ := os.ReadFile(src)
	if err := os.WriteFile(dest, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return dest
}

func TestStagedIn(t *testing.T) {
	dir := t.TempDir()
	touch := func(name string) {
		os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644)
	}
	touch("single-Q4_K_M.gguf")
	touch("split-Q4-00001-of-00003.gguf")
	touch("split-Q4-00002-of-00003.gguf")
	// part 3 missing -> incomplete

	complete := StagedIn(dir, true)
	if len(complete) != 1 || !strings.HasSuffix(complete[0], "single-Q4_K_M.gguf") {
		t.Errorf("complete = %v", complete)
	}
	loose := StagedIn(dir, false)
	if len(loose) != 2 {
		t.Errorf("loose = %v", loose)
	}

	touch("split-Q4-00003-of-00003.gguf")
	complete2 := StagedIn(dir, true)
	if len(complete2) != 2 {
		t.Errorf("after completing split: %v", complete2)
	}
	// The split surfaces once, by its first part.
	for _, p := range complete2 {
		if m := SplitPartRE.FindStringSubmatch(p); m != nil && m[1] != "00001" {
			t.Errorf("continuation part surfaced: %s", p)
		}
	}
}

func TestGeneratePresetsAndReadBack(t *testing.T) {
	t.Setenv("VEGA_HOME", t.TempDir())
	modelsDir := ModelsDir()
	os.MkdirAll(modelsDir, 0o755)
	writeTestGGUF(t, modelsDir, "tiny-Q4_K_M.gguf")

	budget := HardwareBudget{UsableVRAMBytes: 24 * gib, RAMAvailableBytes: 32 * gib}
	presetPath := filepath.Join(RuntimesRoot(), "presets.ini")
	entries, err := GeneratePresets(modelsDir, budget, presetPath, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Refusal != "" {
		t.Fatalf("entries: %+v", entries)
	}
	e := entries[0]
	if e.ModelID != "tiny-Q4_K_M" {
		t.Errorf("model id = %q", e.ModelID)
	}
	// Tiny model on a big card: native window, no spill.
	if e.Window != 262144 || e.Spilled {
		t.Errorf("window=%d spilled=%v", e.Window, e.Spilled)
	}
	// GGUF sampling metadata deferred into the preset.
	if e.Keys["temp"] != "0.7" {
		t.Errorf("sampling keys: %v", e.Keys)
	}
	if e.Keys["cache-type-k"] != "q8_0" || e.Keys["flash-attn"] != "on" {
		t.Errorf("policy keys: %v", e.Keys)
	}

	back := ReadPresetDecisions(presetPath)
	if got := back["tiny-Q4_K_M"]; got == nil || got.Window != 262144 || got.Spilled {
		t.Errorf("read-back: %+v", back)
	}
}

func TestRestoreGrownWindow(t *testing.T) {
	t.Setenv("VEGA_HOME", t.TempDir())
	p := denseProfile()
	budget := HardwareBudget{UsableVRAMBytes: 24 * gib, RAMAvailableBytes: 64 * gib}
	d, _ := InitialWindow(p, budget, true, 0)

	// No override: unchanged.
	if got := restoreGrownWindow("m", p, budget, d, 0); got.Window != d.Window {
		t.Errorf("no override: %d", got.Window)
	}
	// Override above the launch decision and within physics: restored.
	SaveWindowOverride("m", 147456)
	got := restoreGrownWindow("m", p, budget, d, 0)
	if got.Window != 147456 {
		t.Errorf("override: %d", got.Window)
	}
	// Override beyond physics on a smaller-VRAM day: honestly refit down.
	tight := HardwareBudget{UsableVRAMBytes: 17 * gib}
	if got := restoreGrownWindow("m", p, tight, d, 0); got.Window != d.Window {
		t.Errorf("tight refit: %d", got.Window)
	}
	// Clear drops it.
	ClearWindowOverride("m")
	if got := restoreGrownWindow("m", p, budget, d, 0); got.Window != d.Window {
		t.Errorf("cleared: %d", got.Window)
	}
}
