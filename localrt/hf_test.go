package localrt

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// remoteGGUFBytes is a synthetic GGUF header followed by fake tensor data;
// a correct probe parses the header without ever reading the tail.
func remoteGGUFBytes() []byte {
	b := &ggufBuilder{}
	b.str("general.architecture", "qwen3").
		u32("qwen3.block_count", 4).
		u32("qwen3.context_length", 32768).
		u32("qwen3.embedding_length", 1024).
		u32("qwen3.attention.head_count", 8).
		u32("qwen3.attention.head_count_kv", 2).
		f32("general.sampling.temp", 1.0).
		tensor("token_embd.weight", []uint64{1024, 100}, 1)
	return append(b.assemble(), make([]byte, 1<<16)...)
}

// withHFServer points the HF URL templates at a fake server for the test.
func withHFServer(t *testing.T, handler http.Handler) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(handler)
	oldResolve, oldTree := hfResolveURL, hfTreeURL
	hfResolveURL = srv.URL + "/%s/resolve/main/%s"
	hfTreeURL = srv.URL + "/api/models/%s/tree/main?recursive=true"
	t.Cleanup(func() {
		hfResolveURL, hfTreeURL = oldResolve, oldTree
		srv.Close()
	})
	return srv
}

func TestProbeRemoteGGUF(t *testing.T) {
	var gotPath string
	withHFServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Write(remoteGGUFBytes())
	}))

	h, err := ProbeRemoteGGUF(context.Background(), "unsloth/Qwen3-GGUF", "Qwen3-8B-Q4_K_M.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if want := "/unsloth/Qwen3-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"; gotPath != want {
		t.Errorf("requested %q, want %q", gotPath, want)
	}
	if h.Architecture() != "qwen3" {
		t.Errorf("architecture = %q", h.Architecture())
	}
	if h.NCtxTrain() != 32768 {
		t.Errorf("n_ctx_train = %d", h.NCtxTrain())
	}
	if got := h.SamplingDefaults()["temp"]; got != "1" {
		t.Errorf("sampling temp = %q", got)
	}
	// The probe must feed ProfileFromGGUF without a catalog entry.
	if p := ProfileFromGGUF(h); len(p.Layers) != 4 {
		t.Errorf("profile layers = %d", len(p.Layers))
	}
}

func TestProbeRemoteGGUFErrors(t *testing.T) {
	withHFServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/r/x/resolve/main/notgguf.gguf":
			w.Write([]byte("<html>not a model</html>"))
		default:
			http.NotFound(w, r)
		}
	}))

	if _, err := ProbeRemoteGGUF(context.Background(), "r/x", "notgguf.gguf"); err == nil {
		t.Error("expected error for non-GGUF content")
	}
	if _, err := ProbeRemoteGGUF(context.Background(), "r/x", "missing.gguf"); err == nil {
		t.Error("expected error for 404")
	}
}

func TestListHFFiles(t *testing.T) {
	var srv *httptest.Server
	srv = withHFServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/api/models/org/repo/tree/main") {
			http.NotFound(w, r)
			return
		}
		type node struct {
			Type string `json:"type"`
			Path string `json:"path"`
			Size int64  `json:"size"`
		}
		if r.URL.Query().Get("cursor") == "" {
			w.Header().Set("Link", fmt.Sprintf("<%s%s?cursor=page2&recursive=true>; rel=\"next\"", srv.URL, r.URL.Path))
			json.NewEncoder(w).Encode([]node{
				{Type: "file", Path: "README.md", Size: 10},
				{Type: "directory", Path: "Q8_0", Size: 0},
				{Type: "file", Path: "model-Q4_K_M.gguf", Size: 4000},
			})
			return
		}
		json.NewEncoder(w).Encode([]node{
			{Type: "file", Path: "Q8_0/model-Q8_0.gguf", Size: 8000},
		})
	}))

	files, err := ListHFFiles(context.Background(), "org/repo")
	if err != nil {
		t.Fatal(err)
	}
	want := []HFFile{
		{Path: "README.md", SizeBytes: 10},
		{Path: "model-Q4_K_M.gguf", SizeBytes: 4000},
		{Path: "Q8_0/model-Q8_0.gguf", SizeBytes: 8000},
	}
	if len(files) != len(want) {
		t.Fatalf("got %d files, want %d: %+v", len(files), len(want), files)
	}
	for i := range want {
		if files[i] != want[i] {
			t.Errorf("files[%d] = %+v, want %+v", i, files[i], want[i])
		}
	}
}

func TestListHFFilesMissingRepo(t *testing.T) {
	withHFServer(t, http.NotFoundHandler())
	if _, err := ListHFFiles(context.Background(), "no/such"); err == nil {
		t.Error("expected error for missing repo")
	}
}

func TestHFVariants(t *testing.T) {
	files := []HFFile{
		{Path: "README.md", SizeBytes: 10},
		{Path: "mmproj-BF16.gguf", SizeBytes: 500},
		{Path: "Qwen3-8B-UD-Q4_K_XL.gguf", SizeBytes: 4000},
		{Path: "Q8_0/Qwen3-8B-Q8_0-00002-of-00002.gguf", SizeBytes: 3000},
		{Path: "Q8_0/Qwen3-8B-Q8_0-00001-of-00002.gguf", SizeBytes: 5000},
	}
	vs := HFVariants(files)
	if len(vs) != 2 {
		t.Fatalf("got %d variants: %+v", len(vs), vs)
	}
	// Sorted smallest-first.
	if vs[0].Quant != "UD-Q4_K_XL" || vs[0].SizeBytes() != 4000 {
		t.Errorf("variant[0] = %s (%d bytes)", vs[0].Quant, vs[0].SizeBytes())
	}
	if vs[1].Quant != "Q8_0" || vs[1].SizeBytes() != 8000 {
		t.Errorf("variant[1] = %s (%d bytes)", vs[1].Quant, vs[1].SizeBytes())
	}
	// Split parts in order; model id from part 1 with the suffix stripped.
	if got := vs[1].Files[0].Path; got != "Q8_0/Qwen3-8B-Q8_0-00001-of-00002.gguf" {
		t.Errorf("split part order wrong: first = %s", got)
	}
	if got := vs[1].ModelID(); got != "Qwen3-8B-Q8_0" {
		t.Errorf("split model id = %q", got)
	}
}

func TestHFVariantsNoQuantTag(t *testing.T) {
	vs := HFVariants([]HFFile{{Path: "weird-model.gguf", SizeBytes: 100}})
	if len(vs) != 1 || vs[0].Quant != "weird-model" {
		t.Errorf("fallback grouping: %+v", vs)
	}
}

func TestProbeHFModel(t *testing.T) {
	withHFServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/api/models/org/repo/tree/main"):
			w.Write([]byte(`[
				{"type":"file","path":"m-Q4_K_M-00001-of-00002.gguf","size":4000},
				{"type":"file","path":"m-Q4_K_M-00002-of-00002.gguf","size":3000},
				{"type":"file","path":"m-Q8_0.gguf","size":8000}
			]`))
		case strings.HasSuffix(r.URL.Path, "/resolve/main/m-Q4_K_M-00001-of-00002.gguf"):
			w.Write(remoteGGUFBytes())
		default:
			http.NotFound(w, r)
		}
	}))

	// Ambiguous quant must be an error that names the choices.
	if _, err := ProbeHFModel(context.Background(), "org/repo", ""); err == nil ||
		!strings.Contains(err.Error(), "Q8_0") {
		t.Errorf("ambiguous pull error = %v", err)
	}

	probe, err := ProbeHFModel(context.Background(), "org/repo", "q4_k_m") // case-insensitive
	if err != nil {
		t.Fatal(err)
	}
	if probe.Variant.Quant != "Q4_K_M" || probe.Header.Architecture() != "qwen3" {
		t.Errorf("probe = %s / %s", probe.Variant.Quant, probe.Header.Architecture())
	}
	// Split build: weights priced from listed sizes, not part 1's tensor table.
	if got := probe.Profile().WeightsBytes; got != 7000 {
		t.Errorf("split weights bytes = %d, want 7000", got)
	}
}
