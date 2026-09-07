// Per-model preset generation (--models-preset INI) — the router-side
// carrier for context-policy launch decisions. Port of hermes-agent's
// local_runtime/presets.py (MIT, Nous Research — see NOTICE).

package localrt

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Args list -> INI keys. Flags the policy owns; everything else stays out of
// the preset.
var flagToKey = map[string]string{
	"-c": "ctx-size", "-b": "batch-size", "-ub": "ubatch-size",
	"-ctk": "cache-type-k", "-ctv": "cache-type-v", "-fa": "flash-attn",
	"-ot": "override-tensor", "--spec-type": "spec-type", "--spec-draft-n-max": "spec-draft-n-max",
}

// PresetEntry is the launch decision for one staged model.
type PresetEntry struct {
	ModelID string
	Window  int
	Spilled bool
	Refusal string // non-empty when physics refused the model
	Keys    map[string]string
	order   []string // insertion order for deterministic INI output
}

func (p *PresetEntry) set(k, v string) {
	if p.Keys == nil {
		p.Keys = map[string]string{}
	}
	if _, ok := p.Keys[k]; !ok {
		p.order = append(p.order, k)
	}
	p.Keys[k] = v
}

func (p *PresetEntry) setDefault(k, v string) {
	if p.Keys == nil || p.Keys[k] == "" {
		p.set(k, v)
	}
}

func argsToEntry(entry *PresetEntry, args []string) {
	for i := 0; i < len(args); i++ {
		key, ok := flagToKey[args[i]]
		if !ok {
			continue
		}
		if i+1 < len(args) {
			entry.set(key, args[i+1])
			i++
		}
	}
}

// assetPath returns the on-disk path of a catalog companion asset, or ""
// when it isn't downloaded.
func assetPath(a *AssetFile) string {
	if a == nil {
		return ""
	}
	p := filepath.Join(AssetsDir(), a.LocalName())
	if _, err := os.Stat(p); err != nil {
		return ""
	}
	return p
}

// chooseMTPPosture picks (mtpPrefill, logitsBytes) for an MTP model — window
// first, prefill second: price the launch under both postures and keep
// whichever grants the larger window. Never trade context away for prefill.
func chooseMTPPosture(p *ModelProfile, budget HardwareBudget, fixedOverhead int64) (bool, int64) {
	plainLogits := UbLogitsBytes(p.NVocab, true, false)
	stackedLogits := UbLogitsBytes(p.NVocab, true, true)
	stacked, stackedRefused := InitialWindow(p, budget, true, fixedOverhead+stackedLogits)
	plain, plainRefused := InitialWindow(p, budget, true, fixedOverhead+plainLogits)
	if stackedRefused == nil && !stacked.Spilled() &&
		(plainRefused != nil || stacked.Window >= plain.Window) {
		return true, stackedLogits
	}
	return false, plainLogits
}

// restoreGrownWindow lifts the launch window to where the ladder last grew it
// — capped at native, and only when physics still clears the bigger window on
// THIS boot's budget (a smaller-VRAM day re-fits honestly back down).
func restoreGrownWindow(modelID string, p *ModelProfile, budget HardwareBudget,
	d WindowDecision, overhead int64) WindowDecision {
	override := LoadWindowOverrides()[modelID]
	if override <= d.Window {
		return d
	}
	native := p.NCtxTrain
	if native == 0 {
		native = d.Window
	}
	target := override
	if native < target {
		target = native
	}
	kv := CtxBytes(p, target, true)
	need := p.WeightsBytes + kv + overhead
	if need > budget.UsableVRAMBytes+budget.RAMAvailableBytes {
		return d
	}
	spill := need - budget.UsableVRAMBytes
	if spill < 0 {
		spill = 0
	}
	return WindowDecision{Window: target, SpillBytes: spill,
		KVOnGPU: kv <= budget.UsableVRAMBytes,
		Reasons: []string{fmt.Sprintf("grown window restored (%dK)", target/1024)}}
}

// presetFor computes the launch decision for one staged model, or nil when
// its header is unreadable.
func presetFor(gguf string, budget HardwareBudget, catalog []*CatalogEntry) *PresetEntry {
	modelID := ModelIDFromStem(strings.TrimSuffix(filepath.Base(gguf), ".gguf"))
	header, err := ReadGGUFHeader(gguf)
	if err != nil {
		return nil
	}
	profile := ProfileFromGGUF(header)
	entry, _ := FindEntryForModel(catalog, modelID)
	isMTP := entry != nil && entry.MTP
	if isMTP && profile.KVScale == 1.0 {
		// Header-derived profiles don't know about MTP's draft context; apply
		// the calibrated KV multiplier so the launch fit prices what the
		// server will actually allocate.
		profile.KVScale = 1.2
	}

	var mmprojPath string
	fixedOverhead := RuntimeOverheadBytes
	if entry != nil {
		mmprojPath = assetPath(entry.MMProj)
		if mmprojPath != "" {
			fixedOverhead += entry.MMProj.SizeBytes
		}
	}
	var mtpPrefill bool
	var logitsBytes int64
	if isMTP {
		mtpPrefill, logitsBytes = chooseMTPPosture(profile, budget, fixedOverhead)
	} else {
		logitsBytes = UbLogitsBytes(profile.NVocab, false, false)
	}
	overhead := fixedOverhead + logitsBytes

	decision, refusal := InitialWindow(profile, budget, true, overhead)
	if refusal != nil {
		return &PresetEntry{ModelID: modelID, Refusal: refusal.Message}
	}
	decision = restoreGrownWindow(modelID, profile, budget, decision, overhead)

	out := &PresetEntry{ModelID: modelID, Window: decision.Window, Spilled: decision.Spilled()}
	draftDepth := 3
	if entry != nil && entry.MTPDraftDepth > 0 {
		draftDepth = entry.MTPDraftDepth
	}
	// The launch flags MUST match the pricing above (same entry/isMTP/posture).
	argsToEntry(out, LaunchArgs(profile, decision, LaunchOptions{
		FlashAttention: true, MTPCapable: isMTP, MTPDraftDepth: draftDepth,
		UMA: budget.UMA, MTPPrefill: mtpPrefill}))
	if entry != nil && isMTP {
		// Integrated-MTP targets sample on the backend, and so does the draft.
		out.set("backend-sampling", "on")
		out.set("spec-draft-backend-sampling", "on")
	}

	// Sampling deference ladder, under the policy keys (policy wins on
	// clash): the GGUF's own general.sampling.* metadata is the publisher's
	// recommendation; catalog sampling applies only where the file is
	// silent; a model carrying neither runs llama.cpp defaults.
	sampDefaults := header.SamplingDefaults()
	for _, k := range sortedKeys(sampDefaults) {
		out.setDefault(k, sampDefaults[k])
	}
	if entry != nil {
		for _, k := range sortedKeys(entry.Sampling) {
			out.setDefault(k, entry.Sampling[k])
		}
		if mmprojPath != "" {
			out.set("mmproj", mmprojPath)
		}
		if decision.Spilled() {
			if draftPath := assetPath(entry.Draft); draftPath != "" {
				out.set("model-draft", draftPath)
				out.set("spec-type", "draft-dspark")
				// Measured cliff: acceptance 83% at 2-3 drafts, collapses at 4.
				out.set("spec-draft-n-max", "3")
			}
		}
	}
	return out
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// GeneratePresets walks the staged models, runs the launch decision per
// model, and writes one INI. Refused models get no section (callers surface
// the refusal from the returned entries).
func GeneratePresets(modelsDir string, budget HardwareBudget, presetPath string,
	catalog []*CatalogEntry) ([]*PresetEntry, error) {
	var entries []*PresetEntry
	var sections []string
	for _, gguf := range StagedIn(modelsDir, false) {
		entry := presetFor(gguf, budget, catalog)
		if entry == nil {
			continue
		}
		entries = append(entries, entry)
		if entry.Refusal != "" {
			continue
		}
		var body strings.Builder
		fmt.Fprintf(&body, "[%s]\n", entry.ModelID)
		for _, k := range entry.order {
			fmt.Fprintf(&body, "%s = %s\n", k, entry.Keys[k])
		}
		sections = append(sections, body.String())
	}
	if err := os.MkdirAll(filepath.Dir(presetPath), 0o755); err != nil {
		return entries, err
	}
	if err := os.WriteFile(presetPath, []byte(strings.Join(sections, "\n")), 0o644); err != nil {
		return entries, err
	}
	return entries, nil
}

// ReadPresetDecisions reads back the launch decisions the running server was
// actually given (the INI is the record — it's what spawned the children).
// Missing/unparseable -> empty map.
func ReadPresetDecisions(presetPath string) map[string]*PresetEntry {
	out := map[string]*PresetEntry{}
	raw, err := os.ReadFile(presetPath)
	if err != nil {
		return out
	}
	var current *PresetEntry
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			current = &PresetEntry{ModelID: line[1 : len(line)-1]}
			out[current.ModelID] = current
			continue
		}
		if current == nil || line == "" {
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		current.set(k, v)
		if k == "ctx-size" {
			current.Window, _ = strconv.Atoi(v)
		}
		if k == "override-tensor" {
			current.Spilled = true
		}
	}
	return out
}
