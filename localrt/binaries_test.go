package localrt

import (
	"archive/zip"
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSelectBackend(t *testing.T) {
	cases := []struct {
		vendor, osName, want string
	}{
		{"", "macos", "metal"},
		{"nvidia GeForce RTX 4090", "macos", "metal"}, // macOS is always metal
		{"nvidia GeForce RTX 4090", "win", "cuda"},
		{"nvidia something", "ubuntu", "cuda"},
		{"amd", "ubuntu", "vulkan"},
		{"AMD Radeon RX 7900", "win", "vulkan"},
		{"intel", "win", "vulkan"},
		{"", "ubuntu", "cpu"},
		{"", "win", "cpu"},
	}
	for _, c := range cases {
		if got := SelectBackend(c.vendor, c.osName); got != c.want {
			t.Errorf("SelectBackend(%q, %q) = %q, want %q", c.vendor, c.osName, got, c.want)
		}
	}
}

func TestResolveAssets(t *testing.T) {
	// macOS: unified tarball regardless of backend.
	p, err := ResolveAssets("b10679", "metal", "macos", "arm64")
	if err != nil || len(p.Assets) != 1 || p.Assets[0] != "llama-b10679-bin-macos-arm64.tar.gz" {
		t.Errorf("macos: %+v err=%v", p, err)
	}
	// Windows CUDA pairs runtime + cudart.
	p2, err := ResolveAssets("b10679", "cuda", "win", "x64")
	if err != nil || len(p2.Assets) != 2 ||
		p2.Assets[0] != "llama-b10679-bin-win-cuda-13.3-x64.zip" ||
		p2.Assets[1] != "cudart-llama-bin-win-cuda-13.3-x64.zip" {
		t.Errorf("win cuda: %+v err=%v", p2, err)
	}
	// arm64 Windows CUDA uses the newer CUDA version.
	p3, _ := ResolveAssets("b10679", "cuda", "win", "arm64")
	if !strings.Contains(p3.Assets[0], "cuda-13.4-arm64") {
		t.Errorf("win cuda arm64: %+v", p3)
	}
	// Linux CUDA is honestly unsupported.
	if _, err := ResolveAssets("b10679", "cuda", "ubuntu", "x64"); err == nil {
		t.Error("ubuntu cuda should error")
	}
	// win-vulkan-arm64 has no asset.
	if _, err := ResolveAssets("b10679", "vulkan", "win", "arm64"); err == nil {
		t.Error("win vulkan arm64 should error")
	}
	// Unknown platform.
	if _, err := ResolveAssets("b10679", "cpu", "plan9", "x64"); err == nil {
		t.Error("plan9 should error")
	}
}

// fakeServerZip builds a zip holding an executable llama-server script that
// prints a version containing the tag's number.
func fakeServerZip(t *testing.T, tag string) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	hdr := &zip.FileHeader{Name: "build/bin/llama-server", Method: zip.Deflate}
	hdr.SetMode(0o755)
	w, err := zw.CreateHeader(hdr)
	if err != nil {
		t.Fatal(err)
	}
	script := "#!/bin/sh\necho \"version: " + strings.TrimPrefix(tag, "b") + " (deadbeef)\"\n"
	w.Write([]byte(script))
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestInstallPlanEndToEnd(t *testing.T) {
	t.Setenv("VEGA_HOME", t.TempDir())
	tag := "b10679"
	zipBytes := fakeServerZip(t, tag)

	downloads := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		downloads++
		w.Write(zipBytes)
	}))
	defer ts.Close()
	oldURL := releaseURL
	releaseURL = ts.URL + "/%s/%s"
	defer func() { releaseURL = oldURL }()

	plan := AssetPlan{Tag: tag, Backend: "cpu", Assets: []string{"llama-test.zip"}}
	var stages []string
	installDir, err := installPlan(plan, nil, func(stage string, done, total int64, label string) {
		stages = append(stages, stage)
	})
	if err != nil {
		t.Fatal(err)
	}
	if !ManifestVerified(filepath.Join(installDir, "manifest.json")) {
		t.Fatal("manifest not verified after install")
	}
	exe, err := ServerBinary(installDir)
	if err != nil || !strings.HasSuffix(exe, "llama-server") {
		t.Fatalf("server binary: %q err=%v", exe, err)
	}
	if got := InstalledTags(); len(got) != 1 || got[0] != tag {
		t.Errorf("installed tags = %v", got)
	}
	joined := strings.Join(stages, ",")
	if !strings.Contains(joined, "download") || !strings.Contains(joined, "extract") {
		t.Errorf("progress stages: %v", stages)
	}

	// Idempotent: second call touches nothing.
	before := downloads
	if _, err := installPlan(plan, nil, nil); err != nil {
		t.Fatal(err)
	}
	if downloads != before {
		t.Error("verified install re-downloaded")
	}

	// SHA pin mismatch fails and evicts the cached archive.
	os.RemoveAll(filepath.Join(RuntimesRoot(), tag))
	if _, err := installPlan(plan, map[string]string{"llama-test.zip": "00"}, nil); err == nil {
		t.Fatal("expected sha mismatch error")
	}
	if _, err := os.Stat(filepath.Join(RuntimesRoot(), "downloads", "llama-test.zip")); err == nil {
		t.Error("mismatched archive not evicted")
	}
}

func TestInstallRejectsZipSlip(t *testing.T) {
	t.Setenv("VEGA_HOME", t.TempDir())
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, _ := zw.Create("../../evil")
	w.Write([]byte("nope"))
	zw.Close()

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(buf.Bytes())
	}))
	defer ts.Close()
	oldURL := releaseURL
	releaseURL = ts.URL + "/%s/%s"
	defer func() { releaseURL = oldURL }()

	plan := AssetPlan{Tag: "b1", Backend: "cpu", Assets: []string{"evil.zip"}}
	if _, err := installPlan(plan, nil, nil); err == nil || !strings.Contains(err.Error(), "escapes") {
		t.Fatalf("zip slip not rejected: %v", err)
	}
}

func TestInstalledTagsOrdering(t *testing.T) {
	t.Setenv("VEGA_HOME", t.TempDir())
	for _, tag := range []string{"b9999", "b10679"} {
		dir := filepath.Join(RuntimesRoot(), tag, "cpu")
		os.MkdirAll(dir, 0o755)
		os.WriteFile(filepath.Join(dir, "manifest.json"),
			[]byte(`{"verified_version":"x"}`), 0o644)
	}
	// An unverified tag is invisible.
	os.MkdirAll(filepath.Join(RuntimesRoot(), "b11111", "cpu"), 0o755)

	got := InstalledTags()
	if len(got) != 2 || got[0] != "b10679" || got[1] != "b9999" {
		t.Errorf("tags = %v", got)
	}
}
