package localrt

import "testing"

func testEntry(id string, sizeGiB int64, quality int, decodeFraction float64) *CatalogEntry {
	return &CatalogEntry{
		ID: id, DisplayName: id, Repo: "test/" + id,
		Variants: []QuantVariant{{Quant: "Q4_K_M",
			Files: []AssetFile{{Path: id + "-Q4_K_M.gguf", SizeBytes: sizeGiB * gib}}}},
		NCtxTrain: 262144, FullLayers: 16, PerLayerF16: 4096,
		NVocab: 150000, Quality: quality, DecodeFraction: decodeFraction,
	}
}

func TestPackagedCatalogLoads(t *testing.T) {
	entries := Catalog()
	if len(entries) == 0 {
		t.Fatal("empty packaged catalog")
	}
	for _, e := range entries {
		if e.ID == "" || len(e.Variants) == 0 || e.Variants[0].ModelID() == "" {
			t.Errorf("bad entry: %+v", e)
		}
	}
	// Split-part model ids strip the part suffix.
	e, v := FindEntryForModel(entries, "Qwen3.8-Flash-Next-UD-Q4_K_XL")
	if e == nil || v == nil {
		t.Fatal("split model id not found in packaged catalog")
	}
	if v.SizeBytes() <= v.Files[0].SizeBytes {
		t.Error("split variant size should sum all parts")
	}
}

func TestSelectVariant(t *testing.T) {
	e := testEntry("m", 16, 50, 1.0)

	// Plenty of VRAM: zero-spill at the target window.
	big := HardwareBudget{UsableVRAMBytes: 48 * gib, RAMAvailableBytes: 32 * gib}
	c := SelectVariant(e, big)
	if c == nil || !c.ZeroSpill || c.ReasonKey != "best-large-window" {
		t.Errorf("big: %+v", c)
	}

	// Fits at floor but not target: need = 16 weights + 1.5 overhead + ~1.1
	// logits ≈ 18.6 GiB; +2.1 floor-KV = 20.8 fits 22, +4.8 target-KV = 23.4
	// doesn't.
	mid := HardwareBudget{UsableVRAMBytes: 22 * gib, RAMAvailableBytes: 32 * gib}
	c2 := SelectVariant(e, mid)
	if c2 == nil || !c2.ZeroSpill || c2.ReasonKey != "best-fits" {
		t.Errorf("mid: %+v", c2)
	}

	// Weights exceed VRAM but fit VRAM+RAM: spilled.
	small := HardwareBudget{UsableVRAMBytes: 8 * gib, RAMAvailableBytes: 32 * gib}
	c3 := SelectVariant(e, small)
	if c3 == nil || c3.ZeroSpill || c3.ReasonKey != "smallest-fits-spilled" {
		t.Errorf("small: %+v", c3)
	}

	// Nothing fits: nil.
	if c4 := SelectVariant(e, HardwareBudget{UsableVRAMBytes: 4 * gib, RAMAvailableBytes: 4 * gib}); c4 != nil {
		t.Errorf("tiny: %+v", c4)
	}
}

func TestStarterEntry(t *testing.T) {
	starter := testEntry("small", 3, 40, 1.0)
	starter.Starter = true
	big := testEntry("big", 16, 90, 1.0)
	budget := HardwareBudget{UsableVRAMBytes: 48 * gib, RAMAvailableBytes: 32 * gib}

	// The starter wins even when a higher-quality model also fits.
	e, c := StarterEntry([]*CatalogEntry{big, starter}, budget)
	if e == nil || e.ID != "small" || c == nil || !c.ZeroSpill {
		t.Fatalf("starter pick: %+v %+v", e, c)
	}

	// No starter in the set: nil.
	if e, _ := StarterEntry([]*CatalogEntry{big}, budget); e != nil {
		t.Fatalf("non-starter picked: %+v", e)
	}

	// A starter that would run spilled is no starter — the first chat must
	// be pleasant, not a RAM-streaming crawl.
	spilly := HardwareBudget{UsableVRAMBytes: 2 * gib, RAMAvailableBytes: 32 * gib}
	if e, _ := StarterEntry([]*CatalogEntry{starter}, spilly); e != nil {
		t.Fatalf("spilled starter picked: %+v", e)
	}
}

func TestPackagedCatalogStarter(t *testing.T) {
	entries := Catalog()
	var starters []*CatalogEntry
	for _, e := range entries {
		if e.Starter {
			starters = append(starters, e)
		}
	}
	if len(starters) != 1 {
		t.Fatalf("packaged catalog must ship exactly one starter, got %d", len(starters))
	}
	s := starters[0]

	// The starter's promise: a quick download and a resident fit on a modest
	// machine (12 GiB usable, unified memory).
	if got := s.DownloadBytes(s.Variants[len(s.Variants)-1]); got > 5e9 {
		t.Errorf("starter download %d bytes exceeds the 5 GB first-run budget", got)
	}
	modest := HardwareBudget{UsableVRAMBytes: 12 * gib, RAMAvailableBytes: 8 * gib, UMA: true}
	e, c := StarterEntry(entries, modest)
	if e == nil || e.ID != s.ID || !c.ZeroSpill {
		t.Fatalf("starter does not fit a 12 GiB machine resident: %+v %+v", e, c)
	}
}

func TestRecommendedEntry(t *testing.T) {
	// smart is big and slow on UMA; fast is small; both resident on 64 GiB.
	smart := testEntry("smart", 30, 90, 1.0) // UMA: 210/30 = 7 tok/s -> below pleasant floor
	fast := testEntry("fast", 8, 60, 1.0)    // UMA: 210/8 = 26 tok/s
	budget := HardwareBudget{UsableVRAMBytes: 64 * gib, RAMAvailableBytes: 0, UMA: true}

	rec := RecommendedEntry([]*CatalogEntry{smart, fast}, budget)
	if rec == nil || rec.Entry.ID != "fast" || rec.Reason != "speed-gated-quality" {
		t.Fatalf("uma rec: %+v", rec)
	}

	// On a discrete card both clear the floor: quality wins.
	discrete := HardwareBudget{UsableVRAMBytes: 64 * gib, RAMAvailableBytes: 32 * gib}
	rec2 := RecommendedEntry([]*CatalogEntry{smart, fast}, discrete)
	if rec2 == nil || rec2.Entry.ID != "smart" || rec2.Reason != "best-quality-resident" {
		t.Fatalf("discrete rec: %+v", rec2)
	}

	// MoE spilled beats dense spilled when nothing is resident.
	moe := testEntry("moe", 40, 70, 0.1)
	moe.MoE = true
	dense := testEntry("dense", 40, 95, 1.0)
	tiny := HardwareBudget{UsableVRAMBytes: 8 * gib, RAMAvailableBytes: 48 * gib}
	rec3 := RecommendedEntry([]*CatalogEntry{dense, moe}, tiny)
	if rec3 == nil || rec3.Entry.ID != "moe" || rec3.Reason != "least-painful-spilled" {
		t.Fatalf("spilled rec: %+v", rec3)
	}

	// Nothing fits at all.
	if rec4 := RecommendedEntry([]*CatalogEntry{smart}, HardwareBudget{UsableVRAMBytes: gib}); rec4 != nil {
		t.Fatalf("nofit rec: %+v", rec4)
	}
}
