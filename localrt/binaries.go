// Binary acquisition for the managed llama.cpp runtime. Port of
// hermes-agent's local_runtime/binaries.py (MIT, Nous Research — see NOTICE).

package localrt

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

// DefaultRuntimeTag is the pinned llama.cpp release tag; bumped by proxima
// releases after validation. (Same tag hermes-agent currently ships.)
const DefaultRuntimeTag = "b10679"

// releaseURL is a var so tests can point at a fake release server.
var releaseURL = "https://github.com/ggml-org/llama.cpp/releases/download/%s/%s"

// Windows CUDA zips ship per CUDA major; the runtime zip must be paired with
// its cudart zip so end users need no toolkit.
const (
	winCUDAVersion      = "13.3"
	winCUDAVersionArm64 = "13.4"
)

// BinaryResolutionError means no usable asset combination exists for this
// platform/backend.
type BinaryResolutionError struct{ msg string }

func (e *BinaryResolutionError) Error() string { return e.msg }

func resolutionErrorf(format string, args ...any) error {
	return &BinaryResolutionError{msg: fmt.Sprintf(format, args...)}
}

// AssetPlan is the exact archives one runtime install needs, in extraction
// order.
type AssetPlan struct {
	Tag     string
	Backend string // cuda | metal | vulkan | hip | cpu
	Assets  []string
}

// InstallDir is where this plan's runtime extracts.
func (p AssetPlan) InstallDir() string {
	return filepath.Join(RuntimesRoot(), p.Tag, p.Backend)
}

// ManifestVerified reports whether an install manifest records a
// verified_version (missing/damaged -> false).
func ManifestVerified(manifestPath string) bool {
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		return false
	}
	var m struct {
		VerifiedVersion string `json:"verified_version"`
	}
	return json.Unmarshal(raw, &m) == nil && m.VerifiedVersion != ""
}

func releaseNumber(tag string) int {
	n := 0
	for _, ch := range tag {
		if ch >= '0' && ch <= '9' {
			n = n*10 + int(ch-'0')
		}
	}
	return n
}

// InstalledTags returns tags with a verified install, newest first by release
// number. The boot ladder and the update check both read installed-ness from
// here — one resolver, every caller.
func InstalledTags() []string {
	entries, err := os.ReadDir(RuntimesRoot())
	if err != nil {
		return nil
	}
	var found []string
	for _, entry := range entries {
		if !entry.IsDir() || entry.Name() == "downloads" {
			continue
		}
		manifests, _ := filepath.Glob(filepath.Join(RuntimesRoot(), entry.Name(), "*", "manifest.json"))
		for _, m := range manifests {
			if ManifestVerified(m) {
				found = append(found, entry.Name())
				break
			}
		}
	}
	sort.Slice(found, func(i, j int) bool {
		return releaseNumber(found[i]) > releaseNumber(found[j])
	})
	return found
}

// hostOSArch normalizes (os, arch) to release-asset vocabulary.
func hostOSArch() (string, string) {
	osName := map[string]string{"windows": "win", "darwin": "macos", "linux": "ubuntu"}[runtime.GOOS]
	if osName == "" {
		osName = runtime.GOOS
	}
	arch := "x64"
	if runtime.GOARCH == "arm64" {
		arch = "arm64"
	}
	return osName, arch
}

// SelectBackend picks CUDA if NVIDIA, Metal on macOS, Vulkan if a non-NVIDIA
// GPU is present, else CPU. --list-devices validates post-install; the
// supervisor's touch generation is ground truth.
func SelectBackend(gpuVendor, osName string) string {
	if osName == "" {
		osName, _ = hostOSArch()
	}
	if osName == "macos" {
		return "metal"
	}
	vendor := strings.ToLower(gpuVendor)
	if strings.Contains(vendor, "nvidia") {
		return "cuda"
	}
	if vendor == "amd" || vendor == "intel" ||
		strings.Contains(vendor, "radeon") || strings.Contains(vendor, "arc") {
		return "vulkan"
	}
	return "cpu"
}

// Per-OS asset-name templates ({tag}, {arch}, {cuda_ver} substituted).
// Windows CUDA pairs the runtime zip with its cudart zip; ubuntu ships
// tarballs, win ships zips.
var assetTemplates = map[string]map[string][]string{
	"ubuntu": {
		"vulkan": {"llama-{tag}-bin-ubuntu-vulkan-{arch}.tar.gz"},
		"hip":    {"llama-{tag}-bin-ubuntu-rocm-7.2-{arch}.tar.gz"},
		"cpu":    {"llama-{tag}-bin-ubuntu-{arch}.tar.gz"},
	},
	"win": {
		"cuda": {"llama-{tag}-bin-win-cuda-{cuda_ver}-{arch}.zip",
			"cudart-llama-bin-win-cuda-{cuda_ver}-{arch}.zip"},
		"vulkan": {"llama-{tag}-bin-win-vulkan-x64.zip"},
		"hip":    {"llama-{tag}-bin-win-hip-radeon-x64.zip"},
		"cpu":    {"llama-{tag}-bin-win-cpu-{arch}.zip"},
	},
}

// ResolveAssets composes the asset list for (tag, backend, platform). Errors
// for pairs the release ships no artifact for; callers fall back down the
// ladder cuda -> vulkan -> cpu. Empty osName/arch use the host's.
func ResolveAssets(tag, backend, osName, arch string) (AssetPlan, error) {
	hostOS, hostArch := hostOSArch()
	if osName == "" {
		osName = hostOS
	}
	if arch == "" {
		arch = hostArch
	}
	if osName == "macos" {
		// macOS tarballs are unified (Metal built in).
		return AssetPlan{Tag: tag, Backend: backend,
			Assets: []string{fmt.Sprintf("llama-%s-bin-macos-%s.tar.gz", tag, arch)}}, nil
	}
	templates, ok := assetTemplates[osName]
	if !ok {
		return AssetPlan{}, resolutionErrorf("unsupported platform %s-%s", osName, arch)
	}
	if osName == "ubuntu" && backend == "cuda" {
		// No prebuilt Linux CUDA zips at current tags — Linux CUDA users
		// build from source or use vulkan; the resolver is honest about it.
		return AssetPlan{}, resolutionErrorf("no prebuilt linux CUDA asset at %s; use vulkan/cpu or a source build", tag)
	}
	if osName == "win" && backend == "vulkan" && arch == "arm64" {
		return AssetPlan{}, resolutionErrorf("no win-vulkan-arm64 asset at %s", tag)
	}
	names, ok := templates[backend]
	if !ok {
		return AssetPlan{}, resolutionErrorf("unsupported %s backend %s", osName, backend)
	}
	cudaVer := winCUDAVersion
	if arch == "arm64" {
		cudaVer = winCUDAVersionArm64
	}
	assets := make([]string, len(names))
	for i, t := range names {
		s := strings.ReplaceAll(t, "{tag}", tag)
		s = strings.ReplaceAll(s, "{arch}", arch)
		assets[i] = strings.ReplaceAll(s, "{cuda_ver}", cudaVer)
	}
	return AssetPlan{Tag: tag, Backend: backend, Assets: assets}, nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Progress ticks through long operations: stage is download|extract|verify,
// label distinguishes multi-asset installs ("1/2").
type Progress func(stage string, done, total int64, label string)

var downloadClient = &http.Client{Timeout: 30 * time.Minute}

// downloadRetries bounds consecutive attempts that make NO progress; an
// attempt that grows the .part resets the budget, so a flaky link that
// keeps inching forward never dies, while a dead one gives up promptly.
const downloadRetries = 5

// downloadRetryDelay is the base backoff (doubled per barren attempt, capped
// at 30s). A var so tests shrink it.
var downloadRetryDelay = 2 * time.Second

// downloadFile streams url -> dest, retrying transient failures by resuming
// the .part via Range. HTTP 4xx is permanent and fails immediately.
func downloadFile(url, dest string, progress func(done, total int64)) error {
	tmp := dest + ".part"
	partSize := func() int64 {
		info, err := os.Stat(tmp)
		if err != nil {
			return 0
		}
		return info.Size()
	}
	left, delay := downloadRetries, downloadRetryDelay
	for {
		before := partSize()
		err := downloadFileOnce(url, dest, progress)
		if err == nil {
			return nil
		}
		var status *httpStatusError
		if errors.As(err, &status) && status.code >= 400 && status.code < 500 {
			return err
		}
		if partSize() > before {
			left, delay = downloadRetries, downloadRetryDelay
		} else {
			left--
		}
		if left <= 0 {
			return err
		}
		slog.Warn("download interrupted; resuming", "err", err, "in", delay)
		time.Sleep(delay)
		if delay *= 2; delay > 30*time.Second {
			delay = 30 * time.Second
		}
	}
}

// httpStatusError is a non-2xx download response; 4xx never retries.
type httpStatusError struct {
	url    string
	status string
	code   int
}

func (e *httpStatusError) Error() string { return fmt.Sprintf("GET %s: %s", e.url, e.status) }

// downloadFileOnce is a single attempt: stream url -> dest with a .part temp
// file, resuming any existing partial via Range. progress ticks per chunk
// (total 0 when the server sends no Content-Length) — a several-hundred-MB
// archive must never look hung.
func downloadFileOnce(url, dest string, progress func(done, total int64)) error {
	tmp := dest + ".part"
	var done int64
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	// Resume a previous partial download when the server honors ranges.
	if info, err := os.Stat(tmp); err == nil && info.Size() > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", info.Size()))
		done = info.Size()
	}
	resp, err := downloadClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	flags := os.O_CREATE | os.O_WRONLY
	switch resp.StatusCode {
	case http.StatusPartialContent:
		flags |= os.O_APPEND
	case http.StatusOK:
		done = 0
		flags |= os.O_TRUNC
	default:
		return &httpStatusError{url: url, status: resp.Status, code: resp.StatusCode}
	}
	total := done + resp.ContentLength
	if resp.ContentLength < 0 {
		total = 0
	}
	f, err := os.OpenFile(tmp, flags, 0o644)
	if err != nil {
		return err
	}
	buf := make([]byte, 1<<20)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				f.Close()
				return werr
			}
			done += int64(n)
			if progress != nil {
				progress(done, total)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			f.Close()
			return err
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dest)
}

// safeJoin rejects zip-slip paths escaping dest.
func safeJoin(dest, name string) (string, error) {
	p := filepath.Join(dest, name)
	if !strings.HasPrefix(p, filepath.Clean(dest)+string(os.PathSeparator)) {
		return "", fmt.Errorf("archive member escapes destination: %s", name)
	}
	return p, nil
}

func extractZip(archive, dest string, progress func(done, total int64)) error {
	zr, err := zip.OpenReader(archive)
	if err != nil {
		return err
	}
	defer zr.Close()
	var total, done int64
	for _, f := range zr.File {
		total += int64(f.UncompressedSize64)
	}
	for _, f := range zr.File {
		target, err := safeJoin(dest, f.Name)
		if err != nil {
			return err
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		mode := f.Mode().Perm()
		if mode == 0 {
			mode = 0o644
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
		if err != nil {
			rc.Close()
			return err
		}
		_, err = io.Copy(out, rc)
		rc.Close()
		out.Close()
		if err != nil {
			return err
		}
		done += int64(f.UncompressedSize64)
		if progress != nil {
			progress(done, total)
		}
	}
	return nil
}

func extractTarGz(archive, dest string, progress func(done, total int64)) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	var done int64
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		target, err := safeJoin(dest, hdr.Name)
		if err != nil {
			return err
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			mode := os.FileMode(hdr.Mode).Perm()
			if mode == 0 {
				mode = 0o644
			}
			out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			out.Close()
			done += hdr.Size
			if progress != nil {
				progress(done, 0)
			}
		case tar.TypeSymlink:
			// Runtime tarballs carry relative lib symlinks; reject anything
			// pointing outside the tree.
			if filepath.IsAbs(hdr.Linkname) || strings.Contains(hdr.Linkname, "..") {
				continue
			}
			os.Remove(target)
			if err := os.Symlink(hdr.Linkname, target); err != nil {
				return err
			}
		}
	}
}

func extractArchive(archive, dest string, progress func(done, total int64)) error {
	if strings.HasSuffix(archive, ".zip") {
		return extractZip(archive, dest, progress)
	}
	return extractTarGz(archive, dest, progress)
}

// ServerBinary locates llama-server within an extracted runtime (archives
// differ in whether they nest a build/bin directory).
func ServerBinary(installDir string) (string, error) {
	names := []string{"llama-server.exe", "llama-server"}
	for _, name := range names {
		direct := filepath.Join(installDir, name)
		if _, err := os.Stat(direct); err == nil {
			return direct, nil
		}
	}
	var hits []string
	for _, name := range names {
		filepath.WalkDir(installDir, func(path string, d os.DirEntry, err error) error {
			if err == nil && !d.IsDir() && d.Name() == name {
				hits = append(hits, path)
			}
			return nil
		})
		if len(hits) > 0 {
			sort.Strings(hits)
			return hits[0], nil
		}
	}
	return "", resolutionErrorf("llama-server not found under %s", installDir)
}

// VerifyInstall runs --version and requires the tag's build number in the
// output (printed WITHOUT the 'b').
func VerifyInstall(installDir, tag string) (string, error) {
	exe, err := ServerBinary(installDir)
	if err != nil {
		return "", err
	}
	cmd := exec.Command(exe, "--version")
	cmd.Dir = filepath.Dir(exe)
	out, _ := cmd.CombinedOutput()
	text := strings.TrimSpace(string(out))
	if !strings.Contains(text, strings.TrimPrefix(tag, "b")) {
		return "", resolutionErrorf("version check failed for %s: expected %s, got: %.120s", exe, tag, text)
	}
	if i := strings.IndexByte(text, '\n'); i >= 0 {
		return text[:i], nil
	}
	return text, nil
}

// PruneOldTags retains only the tags in keep (current + previous — N-1
// rollback). The shared downloads/ archive cache is not a tag and always
// survives.
func PruneOldTags(keep []string) {
	entries, err := os.ReadDir(RuntimesRoot())
	if err != nil {
		return
	}
	keepSet := map[string]bool{"downloads": true}
	for _, k := range keep {
		keepSet[k] = true
	}
	for _, entry := range entries {
		if entry.IsDir() && !keepSet[entry.Name()] {
			os.RemoveAll(filepath.Join(RuntimesRoot(), entry.Name()))
		}
	}
}

// EnsureRuntimeInstalled is idempotent: resolve, download, verify, extract,
// version-check; returns the install dir. expectedSHA256 pins hashes per
// asset; without pins the computed hash is recorded in the manifest (trust
// on first download, verified on every reinstall).
func EnsureRuntimeInstalled(tag, backend string, expectedSHA256 map[string]string,
	progress Progress) (string, error) {
	plan, err := ResolveAssets(tag, backend, "", "")
	if err != nil {
		return "", err
	}
	return installPlan(plan, expectedSHA256, progress)
}

// installPlan runs the download/verify/extract/version-check pipeline for a
// resolved plan (separated so tests can install synthetic plans).
func installPlan(plan AssetPlan, expectedSHA256 map[string]string, progress Progress) (string, error) {
	installDir := plan.InstallDir()
	manifestPath := filepath.Join(installDir, "manifest.json")
	if ManifestVerified(manifestPath) {
		return installDir, nil
	}
	if err := os.MkdirAll(installDir, 0o755); err != nil {
		return "", err
	}
	downloads := filepath.Join(RuntimesRoot(), "downloads")
	if err := os.MkdirAll(downloads, 0o755); err != nil {
		return "", err
	}

	tick := func(stage, label string) func(done, total int64) {
		if progress == nil {
			return nil
		}
		return func(done, total int64) { progress(stage, done, total, label) }
	}

	recorded := map[string]string{}
	for i, asset := range plan.Assets {
		label := ""
		if len(plan.Assets) > 1 {
			label = fmt.Sprintf("%d/%d", i+1, len(plan.Assets))
		}
		archive := filepath.Join(downloads, asset)
		if _, err := os.Stat(archive); err != nil {
			url := fmt.Sprintf(releaseURL, plan.Tag, asset)
			if err := downloadFile(url, archive, tick("download", label)); err != nil {
				return "", err
			}
		}
		if progress != nil {
			progress("verify", 0, 0, label)
		}
		digest, err := fileSHA256(archive)
		if err != nil {
			return "", err
		}
		if expected := expectedSHA256[asset]; expected != "" && digest != expected {
			os.Remove(archive)
			return "", resolutionErrorf("sha256 mismatch for %s: expected %s, got %s", asset, expected, digest)
		}
		recorded[asset] = digest
		if err := extractArchive(archive, installDir, tick("extract", label)); err != nil {
			return "", err
		}
	}

	if progress != nil {
		progress("verify", 0, 0, "")
	}
	version, err := VerifyInstall(installDir, plan.Tag)
	if err != nil {
		return "", err
	}
	manifest, _ := json.MarshalIndent(map[string]any{
		"tag": plan.Tag, "backend": plan.Backend, "assets": recorded,
		"verified_version": version}, "", "  ")
	if err := os.WriteFile(manifestPath, manifest, 0o644); err != nil {
		return "", err
	}
	return installDir, nil
}
