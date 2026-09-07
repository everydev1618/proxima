// Open Hugging Face pull (DESIGN.md Phase 4 spike): list any GGUF repo's
// files, group them into quant variants, and price a model by streaming only
// its header — so the physics refusal runs BEFORE gigabytes are downloaded.
//
// GGUF puts metadata + the tensor table at the front of the file; a plain GET
// that stops reading once the header is parsed costs a few MB of transfer
// even on a 50GB model. No Range math, no size guessing: the body is closed
// as soon as parseGGUFHeader returns.

package localrt

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path"
	"regexp"
	"sort"
	"strings"
	"time"
)

var (
	// hfTreeURL lists a repo's files with exact sizes; paginated via Link
	// headers. A var so tests can point at a fake server.
	hfTreeURL = "https://huggingface.co/api/models/%s/tree/main?recursive=true"

	// hfClient: generous total timeout — tokenizer-heavy headers run tens of
	// MB and probing must survive a slow line, but never a hung one.
	hfClient = &http.Client{Timeout: 10 * time.Minute}

	// probeMaxHeaderBytes bounds how much of a hostile remote file the header
	// parser may consume before we call it not-a-header.
	probeMaxHeaderBytes = int64(512 << 20)

	// quantRE recognizes quant tags in GGUF filenames (Q4_K_M, UD-Q4_K_XL,
	// IQ4_XS, Q8_0, BF16, MXFP4...). HF convention is uppercase.
	quantRE = regexp.MustCompile(`(?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)+|BF16|F16|F32|MXFP4(?:_[A-Z0-9]+)*)`)
)

// HFFile is one file in a Hugging Face repo tree.
type HFFile struct {
	Path      string
	SizeBytes int64
}

var linkNextRE = regexp.MustCompile(`<([^>]+)>\s*;\s*rel="next"`)

// ListHFFiles fetches a repo's full file tree (paths + exact sizes),
// following pagination.
func ListHFFiles(ctx context.Context, repo string) ([]HFFile, error) {
	url := fmt.Sprintf(hfTreeURL, repo)
	var out []HFFile
	for url != "" {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return nil, err
		}
		resp, err := hfClient.Do(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return nil, fmt.Errorf("listing %s: HTTP %s (private, gated, or no such repo?)", repo, resp.Status)
		}
		var page []struct {
			Type string `json:"type"`
			Path string `json:"path"`
			Size int64  `json:"size"`
		}
		err = json.NewDecoder(resp.Body).Decode(&page)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("listing %s: %w", repo, err)
		}
		for _, f := range page {
			if f.Type == "file" {
				out = append(out, HFFile{Path: f.Path, SizeBytes: f.Size})
			}
		}
		url = ""
		if m := linkNextRE.FindStringSubmatch(resp.Header.Get("Link")); m != nil {
			url = m[1]
		}
	}
	return out, nil
}

// ProbeRemoteGGUF streams just the header of a GGUF hosted on Hugging Face.
func ProbeRemoteGGUF(ctx context.Context, repo, filePath string) (*GGUFHeader, error) {
	url := fmt.Sprintf(hfResolveURL, repo, filePath)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := hfClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("probing %s/%s: HTTP %s", repo, filePath, resp.Status)
	}
	r := bufio.NewReaderSize(io.LimitReader(resp.Body, probeMaxHeaderBytes), 1<<20)
	return parseGGUFHeader(r, repo+"/"+filePath)
}

// HFVariants groups a repo's GGUF weight files into quant variants: split
// parts merge into one variant (part order preserved), companions (mmproj)
// are excluded, and the result is sorted smallest-first. Files without a
// recognizable quant tag group under their own stem.
func HFVariants(files []HFFile) []QuantVariant {
	groups := map[string][]AssetFile{}
	for _, f := range files {
		base := path.Base(f.Path)
		if !strings.HasSuffix(base, ".gguf") || strings.HasPrefix(base, "mmproj") {
			continue
		}
		stem := ModelIDFromStem(strings.TrimSuffix(base, ".gguf"))
		quant := quantRE.FindString(stem)
		if quant == "" {
			quant = stem
		}
		groups[quant] = append(groups[quant], AssetFile{Path: f.Path, SizeBytes: f.SizeBytes})
	}
	out := make([]QuantVariant, 0, len(groups))
	for quant, fs := range groups {
		sort.Slice(fs, func(i, j int) bool { return fs[i].Path < fs[j].Path })
		out = append(out, QuantVariant{Quant: quant, Files: fs})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].SizeBytes() < out[j].SizeBytes() })
	return out
}

// HFModelProbe is everything learned about one HF model before download.
type HFModelProbe struct {
	Repo    string
	Variant QuantVariant
	Header  *GGUFHeader // from the first weight part
}

// Profile builds the pricing profile for the probed variant. Split builds
// carry only part 1's tensor table, so their weights are priced from the
// listed file sizes instead (≈ tensors + a <2% header, slightly
// conservative).
func (p *HFModelProbe) Profile() *ModelProfile {
	prof := ProfileFromGGUF(p.Header)
	if len(p.Variant.Files) > 1 {
		prof.WeightsBytes = p.Variant.SizeBytes()
	}
	return prof
}

// ProbeHFModel lists a repo, picks the variant matching quant
// (case-insensitive; the sole variant when quant is empty), and streams its
// header. Ambiguity is an error that names the available quants — open pulls
// never guess.
func ProbeHFModel(ctx context.Context, repo, quant string) (*HFModelProbe, error) {
	files, err := ListHFFiles(ctx, repo)
	if err != nil {
		return nil, err
	}
	variants := HFVariants(files)
	if len(variants) == 0 {
		return nil, fmt.Errorf("no GGUF weights in %s", repo)
	}
	quants := make([]string, len(variants))
	for i, v := range variants {
		quants[i] = v.Quant
	}
	var pick *QuantVariant
	if quant == "" {
		if len(variants) != 1 {
			return nil, fmt.Errorf("%s has %d quants (%s): name one, e.g. hf:%s:%s",
				repo, len(variants), strings.Join(quants, ", "), repo, variants[0].Quant)
		}
		pick = &variants[0]
	} else {
		for i := range variants {
			if strings.EqualFold(variants[i].Quant, quant) {
				pick = &variants[i]
				break
			}
		}
		if pick == nil {
			return nil, fmt.Errorf("quant %q not in %s (have %s)", quant, repo, strings.Join(quants, ", "))
		}
	}
	h, err := ProbeRemoteGGUF(ctx, repo, pick.Files[0].Path)
	if err != nil {
		return nil, err
	}
	return &HFModelProbe{Repo: repo, Variant: *pick, Header: h}, nil
}
