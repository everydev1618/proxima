package localrt

import (
	"strings"
	"testing"
)

const gib = int64(1 << 30)

// denseProfile: 32 full-attention layers at 4096 KV bytes/token f16, weights
// 16 GiB, native 256K. per-token f16 = 131072 bytes -> q8 factor 0.53125 ->
// ~69632 bytes/token.
func denseProfile() *ModelProfile {
	layers := make([]Layer, 32)
	for i := range layers {
		layers[i] = Layer{Kind: LayerFull, PerTokenKVF16: 4096}
	}
	return &ModelProfile{Name: "dense-test", WeightsBytes: 16 * gib,
		NCtxTrain: 262144, Layers: layers, KVScale: 1.0}
}

func TestCtxBytes(t *testing.T) {
	p := &ModelProfile{Layers: []Layer{
		{Kind: LayerFull, PerTokenKVF16: 4096},
		{Kind: LayerSWA, PerTokenKVF16: 4096},
		{Kind: LayerRecurrent},
	}, SWAWindow: 1024, KVScale: 1.0}

	// q8 factor = 34/32/2 = 0.53125.
	// full: 4096*0.53125*10000 = 21,760,000
	// swa: capped at 1024 -> 4096*0.53125*1024 = 2,228,224
	// recurrent: 4 MiB = 4,194,304
	want := int64(21760000 + 2228224 + 4194304)
	if got := CtxBytes(p, 10000, true); got != want {
		t.Errorf("CtxBytes = %d, want %d", got, want)
	}
	// f16 (no FA) doubles the attention share.
	wantF16 := int64(40960000 + 4194304*1 + 4096*1024*1)
	if got := CtxBytes(p, 10000, false); got != wantF16 {
		t.Errorf("CtxBytes f16 = %d, want %d", got, wantF16)
	}
	// KVScale multiplies everything.
	p.KVScale = 1.2
	if got := CtxBytes(p, 10000, true); got != int64(float64(want)*1.2) {
		t.Errorf("CtxBytes scaled = %d", got)
	}
}

func TestPhysicsCheck(t *testing.T) {
	p := denseProfile()
	tiny := HardwareBudget{UsableVRAMBytes: 4 * gib, RAMAvailableBytes: 4 * gib}
	refusal := PhysicsCheck(p, tiny, Floor, true)
	if refusal == nil {
		t.Fatal("expected physics refusal on 8 GiB total")
	}
	if !strings.Contains(refusal.Message, "smaller quant") {
		t.Errorf("refusal message: %s", refusal.Message)
	}
	big := HardwareBudget{UsableVRAMBytes: 24 * gib, RAMAvailableBytes: 32 * gib}
	if r := PhysicsCheck(p, big, Floor, true); r != nil {
		t.Errorf("unexpected refusal: %s", r.Message)
	}
}

func TestLadder(t *testing.T) {
	rungs := Ladder(262144)
	want := []int{65536, 98304, 147456, 221184, 262144}
	if len(rungs) != len(want) {
		t.Fatalf("ladder = %v", rungs)
	}
	for i := range want {
		if rungs[i] != want[i] {
			t.Fatalf("ladder = %v, want %v", rungs, want)
		}
	}
	// Native below the floor: single rung at native.
	small := Ladder(32768)
	if len(small) != 1 || small[0] != 32768 {
		t.Errorf("small ladder = %v", small)
	}
}

func TestInitialWindow(t *testing.T) {
	p := denseProfile() // ~69632 B/token at q8

	// 24 GiB usable: 16 GiB weights leave 8 GiB for KV. 98304 tokens *
	// 69632 = ~6.4 GiB fits; 147456 * 69632 = ~9.6 GiB doesn't -> 96K rung.
	d, refusal := InitialWindow(p, HardwareBudget{UsableVRAMBytes: 24 * gib, RAMAvailableBytes: 32 * gib}, true, 0)
	if refusal != nil {
		t.Fatal(refusal)
	}
	if d.Window != 98304 || d.Spilled() {
		t.Errorf("24GiB: window=%d spill=%d", d.Window, d.SpillBytes)
	}

	// 12 GiB usable: weights alone exceed VRAM -> floor held, spilled.
	d2, refusal := InitialWindow(p, HardwareBudget{UsableVRAMBytes: 12 * gib, RAMAvailableBytes: 32 * gib}, true, 0)
	if refusal != nil {
		t.Fatal(refusal)
	}
	if d2.Window != Floor || !d2.Spilled() {
		t.Errorf("12GiB: window=%d spill=%d", d2.Window, d2.SpillBytes)
	}

	// Overhead shrinks the zero-spill rung: with 2 GiB overhead the 96K rung
	// (16+2+6.4 > 24) no longer fits -> 64K.
	d3, refusal := InitialWindow(p, HardwareBudget{UsableVRAMBytes: 24 * gib, RAMAvailableBytes: 32 * gib}, true, 2*gib)
	if refusal != nil {
		t.Fatal(refusal)
	}
	if d3.Window != Floor {
		t.Errorf("overhead: window=%d", d3.Window)
	}
}

func TestEvaluateGrowth(t *testing.T) {
	p := denseProfile()
	budget := HardwareBudget{UsableVRAMBytes: 24 * gib, RAMAvailableBytes: 64 * gib}
	base := GrowthInputs{CurrentWindow: 98304, ServerIdle: true, FlashAttention: true}

	// Below occupancy: hold.
	in := base
	in.SessionTokens = 10000
	if d := EvaluateGrowth(p, budget, in); d.Action != GrowthHold {
		t.Errorf("below occupancy: %+v", d)
	}

	// At occupancy, idle, fast: grow to the next rung.
	in.SessionTokens = 90000
	d := EvaluateGrowth(p, budget, in)
	if d.Action != GrowthGrow || d.NextWindow != 147456 {
		t.Errorf("grow: %+v", d)
	}

	// occupancy_confirmed skips gate 1.
	in2 := base
	in2.SessionTokens = 0
	in2.OccupancyConfirmed = true
	if d := EvaluateGrowth(p, budget, in2); d.Action != GrowthGrow {
		t.Errorf("occupancy confirmed: %+v", d)
	}

	// At native: compress.
	in3 := base
	in3.CurrentWindow = 262144
	in3.OccupancyConfirmed = true
	if d := EvaluateGrowth(p, budget, in3); d.Action != GrowthCompressDefault {
		t.Errorf("native: %+v", d)
	}

	// Busy server: hold.
	in4 := base
	in4.SessionTokens = 90000
	in4.ServerIdle = false
	if d := EvaluateGrowth(p, budget, in4); d.Action != GrowthHold {
		t.Errorf("busy: %+v", d)
	}

	// Below the speed floor: compress.
	in5 := base
	in5.SessionTokens = 90000
	in5.MeasuredDecodeTokS = 3.0
	if d := EvaluateGrowth(p, budget, in5); d.Action != GrowthCompressDefault {
		t.Errorf("slow: %+v", d)
	}

	// Next rung exceeds physics: compress.
	tight := HardwareBudget{UsableVRAMBytes: 17 * gib, RAMAvailableBytes: 0}
	in6 := base
	in6.SessionTokens = 90000
	if d := EvaluateGrowth(p, tight, in6); d.Action != GrowthCompressDefault {
		t.Errorf("physics: %+v", d)
	}
}

func TestLaunchArgs(t *testing.T) {
	p := denseProfile()
	d := WindowDecision{Window: 98304}
	args := LaunchArgs(p, d, LaunchOptions{FlashAttention: true})
	joined := strings.Join(args, " ")
	for _, want := range []string{"-c 98304", "-b 2048", "-ub 2048", "-ctk q8_0", "-fa on"} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in %q", want, joined)
		}
	}
	if strings.Contains(joined, "-ot") {
		t.Errorf("unspilled dense got -ot: %q", joined)
	}

	// Spilled MoE on discrete: expert override present.
	moe := &ModelProfile{MoE: true, Layers: []Layer{{Kind: LayerFull, PerTokenKVF16: 2048}}, KVScale: 1.0}
	spilled := WindowDecision{Window: Floor, SpillBytes: gib}
	args2 := strings.Join(LaunchArgs(moe, spilled, LaunchOptions{FlashAttention: true}), " ")
	if !strings.Contains(args2, "-ot") || !strings.Contains(args2, "exps") {
		t.Errorf("spilled moe args: %q", args2)
	}
	// UMA: never -ot.
	args3 := strings.Join(LaunchArgs(moe, spilled, LaunchOptions{FlashAttention: true, UMA: true}), " ")
	if strings.Contains(args3, "-ot") {
		t.Errorf("uma got -ot: %q", args3)
	}
	// MTP posture.
	args4 := strings.Join(LaunchArgs(p, d, LaunchOptions{FlashAttention: true, MTPCapable: true, MTPDraftDepth: 2}), " ")
	if !strings.Contains(args4, "--spec-type draft-mtp") || !strings.Contains(args4, "--spec-draft-n-max 2") {
		t.Errorf("mtp args: %q", args4)
	}
}

func TestUbLogitsBytes(t *testing.T) {
	if got := UbLogitsBytes(150000, false, false); got != 2048*150000*4 {
		t.Errorf("plain = %d", got)
	}
	if got := UbLogitsBytes(150000, true, false); got != 512*150000*4*2 {
		t.Errorf("mtp = %d", got)
	}
	if got := UbLogitsBytes(150000, true, true); got != int64(float64(2048*150000*4)*1.5) {
		t.Errorf("mtp prefill = %d", got)
	}
}

func TestProfileFromGGUF(t *testing.T) {
	b := &ggufBuilder{}
	b.str("general.architecture", "gemma3").
		u32("gemma3.block_count", 6).
		u32("gemma3.context_length", 131072).
		u32("gemma3.embedding_length", 2048).
		u32("gemma3.attention.head_count", 16).
		u32("gemma3.attention.head_count_kv", 8).
		u32("gemma3.attention.sliding_window", 1024).
		tensor("token_embd.weight", []uint64{2048, 1000}, 1)
	h, err := ReadGGUFHeader(b.write(t))
	if err != nil {
		t.Fatal(err)
	}
	p := ProfileFromGGUF(h)
	// gemma3: 5/6 of attention layers are SWA -> 5 SWA + 1 full.
	nSWA, nFull := 0, 0
	for _, l := range p.Layers {
		switch l.Kind {
		case LayerSWA:
			nSWA++
		case LayerFull:
			nFull++
		}
	}
	if nSWA != 5 || nFull != 1 {
		t.Errorf("gemma3 split: swa=%d full=%d", nSWA, nFull)
	}
	// per-token: 8 heads * (128+128) * 2 = 4096
	if p.Layers[0].PerTokenKVF16 != 4096 {
		t.Errorf("per-token = %d", p.Layers[0].PerTokenKVF16)
	}
	if p.SWAWindow != 1024 || p.NCtxTrain != 131072 || p.MoE {
		t.Errorf("profile: %+v", p)
	}
}
