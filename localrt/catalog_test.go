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
