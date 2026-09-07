package localrt

import (
	"context"
	"os"
	"testing"
	"time"
)

// TestLiveHFProbe hits the real huggingface.co (opt-in: PROXIMA_LIVE_HF=1).
// It validates the Phase 4 thesis end to end: an uncurated repo is listed,
// grouped, and priced from a streamed header in seconds, without downloading
// the weights.
func TestLiveHFProbe(t *testing.T) {
	if os.Getenv("PROXIMA_LIVE_HF") == "" {
		t.Skip("set PROXIMA_LIVE_HF=1 to hit huggingface.co")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	repo := "unsloth/Qwen3.5-4B-GGUF" // also the curated starter, so facts are cross-checkable
	probe, err := ProbeHFModel(ctx, repo, "UD-Q4_K_XL")
	if err != nil {
		t.Fatal(err)
	}
	h := probe.Header
	t.Logf("arch=%s layers=%d n_ctx_train=%d vocab=%d weights=%.1fGB sampling=%v",
		h.Architecture(), h.NLayer(), h.NCtxTrain(), h.NVocab(),
		float64(probe.Profile().WeightsBytes)/1e9, h.SamplingDefaults())
	if h.NLayer() == 0 || h.NCtxTrain() == 0 {
		t.Errorf("header missing estimator inputs: %+v", h.Metadata["general.architecture"])
	}
	// Cross-check against the curated catalog entry for the same model.
	e := EntryByID(Catalog(), "qwen3.5-4b")
	if e != nil && h.NCtxTrain() != e.NCtxTrain {
		t.Errorf("probed n_ctx_train %d != catalog %d", h.NCtxTrain(), e.NCtxTrain)
	}
}
