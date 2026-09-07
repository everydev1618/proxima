// Curated starter catalog for the managed local runtime. Port of
// hermes-agent's local_runtime/catalog.py (MIT, Nous Research — see NOTICE);
// catalog.json data is theirs verbatim, plus proxima's own qwen3.5-4b
// starter entry (estimator inputs derived from Qwen/Qwen3.5-4B config.json).
//
// Every entry carries the estimator inputs (measured on real GGUFs) so the
// picker can price a model BEFORE the user downloads gigabytes; once a file
// is on disk, ProfileFromGGUF is the authority.

package localrt

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"path"
	"strings"
)

//go:embed catalog.json
var packagedCatalog []byte

const catalogSchemaVersion = 1

// AssetFile is one downloadable file: repo-relative path and exact bytes
// (feeds the estimator and the progress bar). Local overrides the on-disk
// name (repos reuse generic names like mmproj-BF16.gguf).
type AssetFile struct {
	Path      string `json:"path"`
	SizeBytes int64  `json:"size_bytes"`
	Local     string `json:"local,omitempty"`
}

// LocalName is the on-disk file name for this asset.
func (a AssetFile) LocalName() string {
	if a.Local != "" {
		return a.Local
	}
	return path.Base(a.Path)
}

// QuantVariant is one downloadable build. Split GGUFs list every part in
// Files; the model loads from the first part.
type QuantVariant struct {
	Quant     string      `json:"quant"`
	Files     []AssetFile `json:"files"`
	Validated bool        `json:"validated,omitempty"`
}

// ModelID derives the servable id from the first file's stem.
func (v QuantVariant) ModelID() string {
	name := path.Base(v.Files[0].Path)
	return ModelIDFromStem(strings.TrimSuffix(name, ".gguf"))
}

// SizeBytes is the total download size of the variant's GGUF parts.
func (v QuantVariant) SizeBytes() int64 {
	var n int64
	for _, f := range v.Files {
		n += f.SizeBytes
	}
	return n
}

// WeightsBytes is the pre-download weights estimate: GGUF bytes ≈ tensor
// bytes + a <2% header — slightly conservative until ProfileFromGGUF reads
// the real table.
func (v QuantVariant) WeightsBytes() int64 { return v.SizeBytes() }

// CatalogEntry is one curated model family.
type CatalogEntry struct {
	ID          string         `json:"id"`
	DisplayName string         `json:"display_name"`
	Description string         `json:"description"`
	Repo        string         `json:"repo"`
	Variants    []QuantVariant `json:"variants"`
	// Estimator inputs (measured or config-derived; quant changes weights,
	// never KV). The GGUF header is the authority after download.
	NCtxTrain       int               `json:"n_ctx_train"`
	FullLayers      int               `json:"full_layers"`
	RecurrentLayers int               `json:"recurrent_layers"`
	PerLayerF16     int               `json:"per_layer_f16"` // KV bytes/token per full-attention layer
	SWALayers       int               `json:"swa_layers,omitempty"`
	SWAWindow       int               `json:"swa_window,omitempty"`
	MoE             bool              `json:"moe,omitempty"`
	MTP             bool              `json:"mtp,omitempty"` // ships MTP heads (spec decode when loaded)
	MTPDraftDepth   int               `json:"mtp_draft_depth,omitempty"`
	NVocab          int               `json:"n_vocab,omitempty"`
	MMProj          *AssetFile        `json:"mmproj,omitempty"` // vision projector
	Draft           *AssetFile        `json:"draft,omitempty"`  // spec-decode draft model
	Sampling        map[string]string `json:"sampling,omitempty"`
	MinEngine       string            `json:"min_engine,omitempty"`
	// Quality is an editorial ordering (higher = smarter), authored once at
	// catalog time. Ranks entries for the per-machine recommendation; never
	// displayed as a score.
	Quality int `json:"quality,omitempty"`
	// Starter marks the small, chatty model the first run bootstraps so a
	// user with no server and no downloads can chat immediately. The catalog
	// ships exactly one.
	Starter bool `json:"starter,omitempty"`
	// DecodeFraction is the fraction of the build's bytes read per decoded
	// token: 1.0 for dense, the active slice for MoE. With memory bandwidth
	// this predicts decode speed.
	DecodeFraction float64 `json:"decode_fraction,omitempty"`
}

// Profile builds the pricing profile for one variant of this entry.
func (e *CatalogEntry) Profile(v QuantVariant) *ModelProfile {
	layers := make([]Layer, 0, e.FullLayers+e.SWALayers+e.RecurrentLayers)
	for i := 0; i < e.FullLayers; i++ {
		layers = append(layers, Layer{Kind: LayerFull, PerTokenKVF16: e.PerLayerF16})
	}
	for i := 0; i < e.SWALayers; i++ {
		layers = append(layers, Layer{Kind: LayerSWA, PerTokenKVF16: e.PerLayerF16})
	}
	for i := 0; i < e.RecurrentLayers; i++ {
		layers = append(layers, Layer{Kind: LayerRecurrent})
	}
	kvScale := 1.0
	if e.MTP {
		kvScale = 1.2
	}
	return &ModelProfile{Name: v.ModelID(), WeightsBytes: v.WeightsBytes(),
		NCtxTrain: e.NCtxTrain, Layers: layers, SWAWindow: e.SWAWindow,
		MoE: e.MoE, NVocab: e.NVocab, KVScale: kvScale}
}

// DownloadFiles returns everything a download job fetches for a variant, in
// order.
func (e *CatalogEntry) DownloadFiles(v QuantVariant) []AssetFile {
	files := append([]AssetFile{}, v.Files...)
	if e.MMProj != nil {
		files = append(files, *e.MMProj)
	}
	if e.Draft != nil {
		files = append(files, *e.Draft)
	}
	return files
}

// DownloadBytes is the total download size for a variant plus companions.
func (e *CatalogEntry) DownloadBytes(v QuantVariant) int64 {
	var n int64
	for _, f := range e.DownloadFiles(v) {
		n += f.SizeBytes
	}
	return n
}

// VariantChoice says which build this machine should download and why.
// ReasonKey is a UI-copy discriminator, not display text.
type VariantChoice struct {
	Variant   QuantVariant
	ZeroSpill bool
	ReasonKey string // "best-large-window" | "best-fits" | "smallest-fits-spilled"
}

// SelectVariant fits the entry's one Q4-class build to this machine;
// headroom buys a bigger window, never a bigger quant. Nil: physics refuses.
func SelectVariant(e *CatalogEntry, budget HardwareBudget) *VariantChoice {
	overhead := RuntimeOverheadBytes + UbLogitsBytes(e.NVocab, e.MTP, false)
	if e.MMProj != nil {
		overhead += e.MMProj.SizeBytes
	}
	native := e.NCtxTrain
	if native == 0 {
		native = Floor
	}
	variant := e.Variants[len(e.Variants)-1]
	profile := e.Profile(variant)
	need := variant.WeightsBytes() + overhead
	vram := budget.UsableVRAMBytes

	target := TargetWindow
	if native < target {
		target = native
	}
	if need+CtxBytes(profile, target, true) <= vram {
		return &VariantChoice{Variant: variant, ZeroSpill: true, ReasonKey: "best-large-window"}
	}
	floor := Floor
	if native < floor {
		floor = native
	}
	floorKV := CtxBytes(profile, floor, true)
	if need+floorKV <= vram {
		return &VariantChoice{Variant: variant, ZeroSpill: true, ReasonKey: "best-fits"}
	}
	if need+floorKV <= vram+budget.RAMAvailableBytes {
		return &VariantChoice{Variant: variant, ZeroSpill: false, ReasonKey: "smallest-fits-spilled"}
	}
	return nil
}

// ── recommendation: best quality that fits and isn't miserably slow ──
//
// Quality is a judgment made once at authoring time (entry.Quality). Speed is
// physics per machine: decode is memory-bound, so predicted tok/s ≈
// bandwidth / bytes-read-per-token. The bandwidth axis is the UMA flag; a
// measured per-machine bandwidth could replace these class constants without
// touching the rule. Predictions order candidates and gate the floor — they
// are not display values.

const (
	discreteBandwidthGBs = 1000.0 // representative GDDR6X/GDDR7 class
	umaBandwidthGBs      = 210.0  // measured on unified-memory hardware
	hostBandwidthGBs     = 80.0   // spilled weights stream over host DRAM

	// PleasantFloorTokS: below this predicted decode speed a model stops
	// feeling pleasant for agentic use. Distinct from the growth policy's
	// 6 tok/s compress floor, which marks unusable, not unpleasant.
	PleasantFloorTokS = 20.0
)

// PredictedDecodeTokS is the memory-bound decode prediction for ordering and
// floor-gating.
func PredictedDecodeTokS(e *CatalogEntry, v QuantVariant, budget HardwareBudget, spilled bool) float64 {
	bandwidth := discreteBandwidthGBs
	if spilled {
		bandwidth = hostBandwidthGBs
	} else if budget.UMA {
		bandwidth = umaBandwidthGBs
	}
	frac := e.DecodeFraction
	if frac == 0 {
		frac = 1.0
	}
	bytesPerToken := float64(v.SizeBytes()) * frac
	if bytesPerToken < 1 {
		bytesPerToken = 1
	}
	return bandwidth * 1e9 / bytesPerToken
}

// Recommendation is the catalog's default pick for this machine.
type Recommendation struct {
	Entry  *CatalogEntry
	Choice *VariantChoice
	// Reason: best-quality-resident | speed-gated-quality |
	// fastest-resident | least-painful-spilled
	Reason string
}

// RecommendedEntry picks the catalog default for this machine, or nil when
// nothing fits.
func RecommendedEntry(entries []*CatalogEntry, budget HardwareBudget) *Recommendation {
	type fit struct {
		e *CatalogEntry
		c *VariantChoice
	}
	var fitting []fit
	for _, e := range entries {
		if c := SelectVariant(e, budget); c != nil {
			fitting = append(fitting, fit{e, c})
		}
	}
	if len(fitting) == 0 {
		return nil
	}
	speed := func(f fit, spilled bool) float64 {
		return PredictedDecodeTokS(f.e, f.c.Variant, budget, spilled)
	}

	var resident, pleasant []fit
	for _, f := range fitting {
		if f.c.ZeroSpill {
			resident = append(resident, f)
			if speed(f, false) >= PleasantFloorTokS {
				pleasant = append(pleasant, f)
			}
		}
	}
	if len(pleasant) > 0 {
		pick := pleasant[0]
		for _, f := range pleasant[1:] {
			if f.e.Quality > pick.e.Quality ||
				(f.e.Quality == pick.e.Quality && f.c.Variant.SizeBytes() < pick.c.Variant.SizeBytes()) {
				pick = f
			}
		}
		reason := "best-quality-resident"
		for _, f := range resident {
			if f.e.Quality > pick.e.Quality {
				reason = "speed-gated-quality"
				break
			}
		}
		return &Recommendation{Entry: pick.e, Choice: pick.c, Reason: reason}
	}
	if len(resident) > 0 {
		pick := resident[0]
		for _, f := range resident[1:] {
			if speed(f, false) > speed(pick, false) {
				pick = f
			}
		}
		return &Recommendation{Entry: pick.e, Choice: pick.c, Reason: "fastest-resident"}
	}
	pick := fitting[0]
	for _, f := range fitting[1:] {
		if speed(f, true) > speed(pick, true) {
			pick = f
		}
	}
	return &Recommendation{Entry: pick.e, Choice: pick.c, Reason: "least-painful-spilled"}
}

// StarterEntry picks the catalog's starter model for a first run, or nil
// when none fits. The starter must fit RESIDENT: a spilled first chat crawls
// and would teach the user that local models are miserable.
func StarterEntry(entries []*CatalogEntry, budget HardwareBudget) (*CatalogEntry, *VariantChoice) {
	for _, e := range entries {
		if !e.Starter {
			continue
		}
		if c := SelectVariant(e, budget); c != nil && c.ZeroSpill {
			return e, c
		}
	}
	return nil, nil
}

// ── catalog loading: packaged JSON only (no network refresh in v1) ──

type catalogDoc struct {
	SchemaVersion int             `json:"schema_version"`
	Models        []*CatalogEntry `json:"models"`
}

// LoadCatalog parses a catalog document, checking the schema version.
func LoadCatalog(raw []byte) ([]*CatalogEntry, error) {
	var doc catalogDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, err
	}
	if doc.SchemaVersion != catalogSchemaVersion {
		return nil, fmt.Errorf("catalog schema %d (this build reads %d)",
			doc.SchemaVersion, catalogSchemaVersion)
	}
	for _, e := range doc.Models {
		if len(e.Variants) == 0 {
			return nil, fmt.Errorf("catalog entry %s has no variants", e.ID)
		}
	}
	return doc.Models, nil
}

// Catalog returns the packaged catalog (parse errors are a build defect and
// panic once at first use).
func Catalog() []*CatalogEntry {
	entries, err := LoadCatalog(packagedCatalog)
	if err != nil {
		panic("packaged catalog.json unreadable: " + err.Error())
	}
	return entries
}

// FindEntryForModel locates the entry + variant that owns a staged model id.
func FindEntryForModel(entries []*CatalogEntry, modelID string) (*CatalogEntry, *QuantVariant) {
	for _, e := range entries {
		for i := range e.Variants {
			if e.Variants[i].ModelID() == modelID {
				return e, &e.Variants[i]
			}
		}
	}
	return nil, nil
}

// EntryByID finds a catalog entry by its family id.
func EntryByID(entries []*CatalogEntry, id string) *CatalogEntry {
	for _, e := range entries {
		if e.ID == id {
			return e
		}
	}
	return nil
}
