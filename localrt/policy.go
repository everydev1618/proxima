// Context policy — the window ladder for managed local models. Port of
// hermes-agent's local_runtime/context_policy.py (MIT, Nous Research — see
// NOTICE).
//
// One contract: any model runs at any window up to its native max; hardware
// and session depth only change tokens/s. Constants, not knobs — nothing in
// this file reads config. The policy encodes behavior measured on real
// hardware (llama.cpp, discrete NVIDIA, unified-memory).

package localrt

import (
	"fmt"
	"regexp"
	"strconv"
)

const (
	// Floor = target; one internal constant.
	Floor            = 64 * 1024
	ladderGrowth     = 1.5
	growAtOccupancy  = 0.85 // of the current window, at turn boundary
	SpeedFloorTokS   = 6.0  // deepest measured spill bottomed near this
	earlyCostCtxFrac = 0.15 // bounded early cost when weights spill

	// TargetWindow is the smallest ladder rung at which compression becomes
	// the exception rather than the routine. Measured over 161 real agentic
	// sessions: 66% complete uncompressed in 64K, 82% in 96K, 91% in 144K —
	// and the marginal gain past 144K falls below the quality cost of
	// stepping down another quant. The Floor remains the guarantee.
	TargetWindow = 144 * 1024

	// RuntimeOverheadBytes is what a load really costs beyond weights + KV:
	// contexts and compute buffers at the default microbatch (-ub 512, no
	// MTP), measured on a 32 GiB card. Microbatch/MTP logits buffers are
	// priced separately per model (UbLogitsBytes). Callers add mmproj bytes
	// on top.
	RuntimeOverheadBytes = int64(1.5 * float64(1<<30))
)

// Ladder returns 64K -> 96K -> 144K -> ... -> native (native always the last
// rung).
func Ladder(native int) []int {
	var rungs []int
	step := float64(Floor)
	for step < float64(native) {
		rungs = append(rungs, int(step))
		step *= ladderGrowth
	}
	return append(rungs, native)
}

// WindowDecision is one launch decision.
type WindowDecision struct {
	Window     int
	SpillBytes int64 // weights displaced to host at this window
	KVOnGPU    bool
	Reasons    []string
}

func (d WindowDecision) Spilled() bool { return d.SpillBytes > 0 }

// InitialWindow is the launch decision: largest cheap rung, never below the
// floor.
//
// Zero-spill rung: weights + ctx + overhead fit usable VRAM entirely.
// Bounded-early-cost rung (weights already exceed VRAM): largest rung whose
// ctx stays <= ~15% of usable VRAM. Floor everywhere, capped at native.
// overheadBytes is runtime cost beyond weights+KV; zero keeps this pure
// physics for decision-table tests, production callers pass it.
func InitialWindow(p *ModelProfile, budget HardwareBudget, flashAttention bool,
	overheadBytes int64) (WindowDecision, *PhysicsRefusal) {
	if refusal := PhysicsCheck(p, budget, Floor, flashAttention); refusal != nil {
		return WindowDecision{}, refusal
	}

	native := p.NCtxTrain
	if native == 0 {
		native = Floor
	}
	rungs := Ladder(native)
	kv := func(rung int) int64 { return CtxBytes(p, rung, flashAttention) }

	bestZeroSpill := 0
	for _, rung := range rungs {
		if p.WeightsBytes+overheadBytes+kv(rung) > budget.UsableVRAMBytes {
			break
		}
		bestZeroSpill = rung
	}

	var window int
	var reason string
	minFloor := Floor
	if native < minFloor {
		minFloor = native
	}
	if bestZeroSpill >= minFloor && bestZeroSpill > 0 {
		window = bestZeroSpill
		reason = fmt.Sprintf("largest zero-spill rung (%dK)", window/1024)
	} else {
		// Weights spill from turn one (steep-curve model on a small card) —
		// hold the floor, bound the early ctx cost.
		cap := int64(float64(budget.UsableVRAMBytes) * earlyCostCtxFrac)
		window = minFloor
		for _, rung := range rungs {
			if rung < window {
				continue
			}
			if kv(rung) > cap {
				break
			}
			window = rung
		}
		reason = fmt.Sprintf("floor held at %dK; weights spill (deliberate price of the guarantee)", window/1024)
	}

	kvBytes := kv(window)
	spill := p.WeightsBytes + kvBytes - budget.UsableVRAMBytes
	if spill < 0 {
		spill = 0
	}
	return WindowDecision{Window: window, Reasons: []string{reason},
		SpillBytes: spill, KVOnGPU: kvBytes <= budget.UsableVRAMBytes}, nil
}

// GrowthAction is what one growth evaluation decided.
type GrowthAction string

const (
	GrowthGrow            GrowthAction = "grow"
	GrowthHold            GrowthAction = "hold"
	GrowthCompressDefault GrowthAction = "compress-default"
)

// GrowthDecision is the outcome of one end-of-turn growth evaluation.
type GrowthDecision struct {
	Action     GrowthAction
	NextWindow int
	Reason     string
}

// GrowthInputs are the live signals one evaluation consumes.
type GrowthInputs struct {
	CurrentWindow      int
	SessionTokens      int
	MeasuredDecodeTokS float64 // 0 = unmeasured
	ServerIdle         bool
	FlashAttention     bool
	// OccupancyConfirmed skips the occupancy gate when the caller's own
	// compression gate already fired, so two edge definitions can't deadlock
	// into compress-before-grow.
	OccupancyConfirmed bool
}

// EvaluateGrowth runs one growth evaluation, END-OF-TURN ONLY (recurrent
// state cannot rewind mid-sequence).
//
// Gate order: occupancy (~85%) -> native cap -> idleness (growth only on an
// otherwise-idle server) -> speed floor (below it compression is the
// default) -> re-fit against LIVE free memory (the rung must fit NOW, not at
// launch).
func EvaluateGrowth(p *ModelProfile, budget HardwareBudget, in GrowthInputs) GrowthDecision {
	if !in.OccupancyConfirmed && float64(in.SessionTokens) < float64(in.CurrentWindow)*growAtOccupancy {
		return GrowthDecision{Action: GrowthHold, Reason: "session below growth occupancy"}
	}

	native := p.NCtxTrain
	if native == 0 {
		native = in.CurrentWindow
	}
	if in.CurrentWindow >= native {
		return GrowthDecision{Action: GrowthCompressDefault, Reason: "at native window; compression is the only move"}
	}
	if !in.ServerIdle {
		return GrowthDecision{Action: GrowthHold, Reason: "server busy; re-grant deferred to idle"}
	}
	if in.MeasuredDecodeTokS > 0 && in.MeasuredDecodeTokS < SpeedFloorTokS {
		return GrowthDecision{Action: GrowthCompressDefault,
			Reason: fmt.Sprintf("decode %.1f tok/s below the ~%.0f tok/s floor; growth is now an explicit per-session choice",
				in.MeasuredDecodeTokS, SpeedFloorTokS)}
	}

	nextRung := native
	for _, r := range Ladder(native) {
		if r > in.CurrentWindow {
			nextRung = r
			break
		}
	}

	// Re-fit against live free memory: allocation beyond residency is the
	// slow path, so a rung that no longer fits doesn't get granted.
	kv := CtxBytes(p, nextRung, in.FlashAttention)
	if p.WeightsBytes+kv > budget.UsableVRAMBytes+budget.RAMAvailableBytes {
		return GrowthDecision{Action: GrowthCompressDefault, Reason: "next rung exceeds physics; compression instead"}
	}
	return GrowthDecision{Action: GrowthGrow, NextWindow: nextRung,
		Reason: fmt.Sprintf("rung %dK -> %dK", in.CurrentWindow/1024, nextRung/1024)}
}

// spillOverrideMoE / spillOverrideHybrid are -ot placements for spilled
// configs: expert/FFN weights to host so attention + KV stay GPU-resident.
var (
	spillOverrideMoE    = `blk\.\d+\.ffn_.*_exps\.weight=CPU`
	spillOverrideHybrid = `blk\.\d+\.ffn_.*\.weight=CPU`
	_                   = regexp.MustCompile(spillOverrideMoE)    // compile-checked
	_                   = regexp.MustCompile(spillOverrideHybrid) // compile-checked
)

// SpillOverrides returns -ot placement args for spilled configs. MoE gets the
// expert pattern; hybrids push recurrent-layer FFNs (their n_head_kv==0
// layers carry no KV worth protecting). Dense: fit's back-to-front layer cut
// is the only axis.
func SpillOverrides(p *ModelProfile) []string {
	if p.MoE {
		return []string{"-ot", spillOverrideMoE}
	}
	if p.RecurrentLayerCount() > 0 {
		return []string{"-ot", spillOverrideHybrid}
	}
	return nil
}

// LaunchOptions parameterize LaunchArgs.
type LaunchOptions struct {
	FlashAttention bool
	MTPCapable     bool
	MTPDraftDepth  int // 0 -> 3
	UMA            bool
	MTPPrefill     bool
}

// LaunchArgs builds per-model launch flags from a window decision. Explicit
// -c puts fit into spill-weights-and-hold-ctx; q8 KV cache wherever flash
// attention exists; -ot placement on spilled configs — DISCRETE cards only.
func LaunchArgs(p *ModelProfile, d WindowDecision, opts LaunchOptions) []string {
	args := []string{"-c", strconv.Itoa(d.Window)}
	if opts.MTPCapable {
		depth := opts.MTPDraftDepth
		if depth == 0 {
			depth = 3
		}
		args = append(args, "--spec-type", "draft-mtp", "--spec-draft-n-max", strconv.Itoa(depth),
			"--backend-sampling", "--spec-draft-backend-sampling")
		if opts.MTPPrefill {
			args = append(args, "-b", "4096", "-ub", "2048")
		}
	} else {
		args = append(args, "-b", "2048", "-ub", "2048")
	}
	if opts.FlashAttention {
		args = append(args, "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on")
	}
	if d.Spilled() && !opts.UMA {
		args = append(args, SpillOverrides(p)...)
	}
	return args
}

// UbLogitsBytes is the GPU logits/compute-buffer cost of the microbatch
// posture chosen by LaunchArgs, priced from the model's own vocab and
// calibrated against measured server RSS.
func UbLogitsBytes(nVocab int, mtpCapable, mtpPrefill bool) int64 {
	v := int64(nVocab)
	if v < 0 {
		v = 0
	}
	if mtpCapable && mtpPrefill {
		return int64(float64(2048*v*4) * 1.5)
	}
	if mtpCapable {
		return 512 * v * 4 * 2
	}
	return 2048 * v * 4
}
