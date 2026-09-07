// Paths and staged-model discovery for the managed runtime. Path conventions
// mirror govega's ~/.vega home (VEGA_HOME override) without importing it, so
// localrt stays dependency-free. Split/staging semantics port hermes-agent's
// local_runtime/bootstrap.py (MIT, Nous Research — see NOTICE).

package localrt

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// VegaHome mirrors govega's home convention: $VEGA_HOME or ~/.vega.
func VegaHome() string {
	if v := os.Getenv("VEGA_HOME"); v != "" {
		return v
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".vega")
}

// RuntimesRoot is where managed llama.cpp builds live. Machine-scoped:
// engine binaries, presets and server state describe this machine's hardware
// and its one managed server.
func RuntimesRoot() string { return filepath.Join(VegaHome(), "runtimes", "llamacpp") }

// ModelsDir holds staged GGUFs. Machine-scoped: a 20 GB GGUF is a machine
// asset, and every process shares the one managed server that serves it.
func ModelsDir() string { return filepath.Join(VegaHome(), "models") }

// AssetsDir holds non-model companion files (mmproj projectors, spec-decode
// drafts). A subdirectory so the router's model listing — and StagedIn —
// never mistakes an asset for a servable model.
func AssetsDir() string { return filepath.Join(ModelsDir(), "assets") }

// StagedIn returns the servable GGUFs in a directory: single files, plus
// split GGUFs once by their first part. With requireComplete a split counts
// only when EVERY part is on disk — a mid-download split is not servable and
// must not surface anywhere as a model.
func StagedIn(dir string, requireComplete bool) []string {
	matches, _ := filepath.Glob(filepath.Join(dir, "*.gguf"))
	sort.Strings(matches)
	names := map[string]bool{}
	for _, p := range matches {
		names[filepath.Base(p)] = true
	}
	var out []string
	for _, p := range matches {
		base := filepath.Base(p)
		m := SplitPartRE.FindStringSubmatch(base)
		if m == nil {
			out = append(out, p)
			continue
		}
		if m[1] != "00001" {
			continue
		}
		if !requireComplete {
			out = append(out, p)
			continue
		}
		stem := base[:strings.LastIndex(base, m[0])]
		total := 0
		fmt.Sscanf(m[2], "%d", &total)
		complete := true
		for i := 2; i <= total; i++ {
			if !names[fmt.Sprintf("%s-%05d-of-%s.gguf", stem, i, m[2])] {
				complete = false
				break
			}
		}
		if complete {
			out = append(out, p)
		}
	}
	return out
}

// StagedModels returns the servable staged models in the default models dir.
func StagedModels() []string { return StagedIn(ModelsDir(), true) }

// StagedModelIDs returns the model ids of the staged models.
func StagedModelIDs() []string {
	paths := StagedModels()
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		out = append(out, ModelIDFromStem(strings.TrimSuffix(filepath.Base(p), ".gguf")))
	}
	return out
}
