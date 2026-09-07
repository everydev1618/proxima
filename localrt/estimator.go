// Per-layer context-memory estimator + physics check. Port of hermes-agent's
// local_runtime/estimator.py (MIT, Nous Research — see NOTICE).
//
// The estimator is ADVISORY: fit's allocation is authoritative at launch and
// the touch generation is ground truth after it. Unknown shapes round UP
// (never underestimate memory).

package localrt

import "fmt"

const (
	// q8_0: 34-byte blocks of 32 f16-equivalent elements (exact).
	q8BytesPerElem  = 34.0 / 32.0
	f16BytesPerElem = 2.0

	// Per-recurrent-layer state allowance (bytes/seq). Deliberately generous:
	// an entire measured hybrid slot state is ~99 MB including 8K tokens of
	// full-attn KV, so tens of MiB total is the right order; unknown SSM
	// shapes must never underestimate.
	recurrentStatePerLayer = 4 << 20
)

// Architectures with a known SWA layer pattern: arch -> fraction of layers
// that are sliding-window. Unknown SWA archs treat every layer as full
// attention (overestimate; safe).
var swaLayerFraction = map[string]float64{"gemma3": 5.0 / 6.0, "gemma2": 1.0 / 2.0}

// LayerKind classifies one transformer layer for KV pricing.
type LayerKind int

const (
	LayerFull LayerKind = iota
	LayerSWA
	LayerRecurrent
)

// Layer is one layer's KV pricing input: kind plus KV bytes/token at f16
// (SWA capped later, recurrent ignored).
type Layer struct {
	Kind          LayerKind
	PerTokenKVF16 int
}

// ModelProfile is everything the policy needs, decoupled from GGUF parsing
// so decision-table tests can construct profiles directly.
type ModelProfile struct {
	Name           string
	WeightsBytes   int64
	EmbdTableBytes int64
	NCtxTrain      int
	Layers         []Layer
	SWAWindow      int
	MoE            bool
	Architecture   string
	NVocab         int // prices logits buffers (ubatch x vocab)
	// KVScale is the context-cost multiplier. MTP spec decode keeps a small
	// draft context beside the main one; calibrated against measured server
	// RSS — the draft adds ~17% to per-token KV; 1.2 rounds up so the error
	// stays on the safe side. 1.0 when unset-equivalent (use NewModelProfile
	// or set explicitly).
	KVScale float64
}

// PerTokenKVF16 is the uncapped per-token KV cost (full + SWA share).
func (p *ModelProfile) PerTokenKVF16() int {
	total := 0
	for _, l := range p.Layers {
		if l.Kind != LayerRecurrent {
			total += l.PerTokenKVF16
		}
	}
	return total
}

// RecurrentLayerCount counts the recurrent/linear layers.
func (p *ModelProfile) RecurrentLayerCount() int {
	n := 0
	for _, l := range p.Layers {
		if l.Kind == LayerRecurrent {
			n++
		}
	}
	return n
}

func (p *ModelProfile) kvScale() float64 {
	if p.KVScale <= 0 {
		return 1.0
	}
	return p.KVScale
}

// HardwareBudget is the memory the physics check may budget against.
// Discrete cards may trust the device query; unified-memory devices must
// budget from OS free memory minus headroom (device queries observed off by
// 3x). Callers construct this accordingly; the estimator just consumes it.
type HardwareBudget struct {
	UsableVRAMBytes   int64 // live free (discrete) / derived (UMA)
	TotalDeviceBytes  int64
	RAMAvailableBytes int64
	UMA               bool
}

// ProfileFromGGUF builds a pricing profile from a parsed header.
func ProfileFromGGUF(h *GGUFHeader) *ModelProfile {
	kvHeads := h.HeadCountsKV()
	dk, dv := h.HeadDimK(), h.HeadDimV()
	swaFraction := swaLayerFraction[h.Architecture()]
	hasSWA := h.SlidingWindow() > 0 && swaFraction > 0

	nAttnTotal := 0
	for _, heads := range kvHeads {
		if heads > 0 {
			nAttnTotal++
		}
	}
	nSWA := 0
	if hasSWA {
		nSWA = int(float64(nAttnTotal)*swaFraction + 0.5)
	}

	layers := make([]Layer, 0, len(kvHeads))
	nAttnSeen := 0
	for _, heads := range kvHeads {
		if heads == 0 {
			layers = append(layers, Layer{Kind: LayerRecurrent})
			continue
		}
		perToken := int(float64(heads)*float64(dk+dv)*f16BytesPerElem + 0.5)
		// Distribute the SWA share across the first nSWA attention layers;
		// only the full/SWA SPLIT matters to the totals, not which indexes.
		kind := LayerFull
		if nAttnSeen < nSWA {
			kind = LayerSWA
		}
		layers = append(layers, Layer{Kind: kind, PerTokenKVF16: perToken})
		nAttnSeen++
	}

	return &ModelProfile{
		Name: h.Path, WeightsBytes: h.TensorBytes, EmbdTableBytes: h.EmbdTableBytes,
		NCtxTrain: h.NCtxTrain(), Layers: layers, SWAWindow: h.SlidingWindow(),
		MoE: h.ExpertCount() > 0, Architecture: h.Architecture(), NVocab: h.NVocab(),
		KVScale: 1.0,
	}
}

// KVDtypeFactor prices the KV cache dtype: q8_0 with flash attention (every
// backend we ship); f16 on exotic non-FA fallbacks — the 64K guarantee stands
// either way, the physics check just prices the doubled KV.
func KVDtypeFactor(flashAttention bool) float64 {
	if flashAttention {
		return q8BytesPerElem / f16BytesPerElem
	}
	return 1.0
}

// CtxBytes is the context memory for one window: full layers linear in T,
// SWA layers capped at the sliding window, recurrent layers constant. Scaled
// by KVScale (MTP draft context).
func CtxBytes(p *ModelProfile, window int, flashAttention bool) int64 {
	factor := KVDtypeFactor(flashAttention)
	total := 0.0
	for _, l := range p.Layers {
		switch l.Kind {
		case LayerRecurrent:
			total += recurrentStatePerLayer
		case LayerSWA:
			w := window
			if p.SWAWindow < w {
				w = p.SWAWindow
			}
			total += float64(l.PerTokenKVF16) * factor * float64(w)
		default:
			total += float64(l.PerTokenKVF16) * factor * float64(window)
		}
	}
	return int64(total * p.kvScale())
}

// PhysicsRefusal is the only true refusal: weights + floor-KV + state exceed
// VRAM + RAM. The remedy is a smaller quant, never a smaller window.
type PhysicsRefusal struct {
	NeededBytes    int64
	AvailableBytes int64
	Message        string
}

func (r *PhysicsRefusal) Error() string { return r.Message }

// PhysicsCheck returns a refusal when the model cannot run at the floor
// window on this budget, else nil.
func PhysicsCheck(p *ModelProfile, budget HardwareBudget, floor int, flashAttention bool) *PhysicsRefusal {
	window := floor
	if p.NCtxTrain > 0 && p.NCtxTrain < window {
		window = p.NCtxTrain
	}
	needed := p.WeightsBytes + CtxBytes(p, window, flashAttention)
	available := budget.UsableVRAMBytes + budget.RAMAvailableBytes
	if needed <= available {
		return nil
	}
	const gib = 1 << 30
	return &PhysicsRefusal{
		NeededBytes: needed, AvailableBytes: available,
		Message: fmt.Sprintf("%s: needs ~%.1f GiB at the %dK floor but only ~%.1f GiB of VRAM+RAM exist — try a smaller quant (UD-Q3/Q2)",
			p.Name, float64(needed)/gib, floor/1024, float64(available)/gib),
	}
}
