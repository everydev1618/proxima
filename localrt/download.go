// Model downloads from Hugging Face into the staged models dir. GGUF parts
// land in ModelsDir; companion assets (mmproj, drafts) land in AssetsDir so
// the router never lists them. Downloads stream to .part files and resume
// via Range, so a multi-GB pull survives an interruption.

package localrt

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// hfResolveURL is a var so tests can point at a fake server.
var hfResolveURL = "https://huggingface.co/%s/resolve/main/%s"

// DownloadModel fetches every file of a catalog entry's variant (weights
// parts + companions). progress ticks per file with stage "download".
func DownloadModel(ctx context.Context, entry *CatalogEntry, variant QuantVariant, progress Progress) error {
	if err := os.MkdirAll(AssetsDir(), 0o755); err != nil {
		return err
	}
	files := entry.DownloadFiles(variant)
	isWeight := map[string]bool{}
	for _, f := range variant.Files {
		isWeight[f.Path] = true
	}
	for i, f := range files {
		label := fmt.Sprintf("%d/%d %s", i+1, len(files), f.LocalName())
		destDir := AssetsDir()
		if isWeight[f.Path] {
			destDir = ModelsDir()
		}
		dest := filepath.Join(destDir, f.LocalName())
		if info, err := os.Stat(dest); err == nil && info.Size() == f.SizeBytes {
			continue // already complete
		}
		url := fmt.Sprintf(hfResolveURL, entry.Repo, f.Path)
		var tick func(done, total int64)
		if progress != nil {
			tick = func(done, total int64) {
				if total == 0 {
					total = f.SizeBytes
				}
				progress("download", done, total, label)
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := downloadFile(url, dest, tick); err != nil {
			return fmt.Errorf("downloading %s: %w", f.Path, err)
		}
	}
	// A model staged after the server booted is invisible to it (the
	// router's model list is spawn-only); callers bounce the managed server.
	return nil
}

// DeleteModel removes a staged model's files (all split parts) and its
// growth state.
func DeleteModel(modelID string) error {
	matches, _ := filepath.Glob(filepath.Join(ModelsDir(), "*.gguf"))
	for _, p := range matches {
		stem := strings.TrimSuffix(filepath.Base(p), ".gguf")
		if ModelIDFromStem(stem) == modelID {
			if err := os.Remove(p); err != nil {
				return err
			}
		}
	}
	return ClearWindowOverride(modelID)
}
